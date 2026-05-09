import torch
import numpy as np
import logging
from scripts import datasets


def get_reconstruction_fn_inverse_propagation(config, initial_measurement,
                                              intermediate_save_indices, delta, device,
                                              share_noise=False):
    """Build an inverse-propagation reconstruction function.

        Args:
            initial_measurement: Initial observation, such as a wide-field image, with the
                    same shape as the training input.
            intermediate_save_indices: Indices of intermediate steps to retain.
            delta: Standard deviation of the noise injected at each step.
            share_noise: Whether to share the same noise realization across the batch.
    """
    K = config.model.K

    def reconstructor(model):
        if share_noise:
            noises = [torch.randn_like(initial_measurement[0], dtype=torch.float)[None]
                      for _ in range(K)]
        intermediate_out = []

        with torch.no_grad():
            u = initial_measurement.to(config.device).float()
            if intermediate_save_indices is not None and K in intermediate_save_indices:
                intermediate_out.append((u, u))
            for i in range(K, 0, -1):
                vec_fwd_steps = torch.ones(
                    initial_measurement.shape[0], device=device, dtype=torch.long) * i
                u_mean = model(u, vec_fwd_steps) + u
                noise = noises[i-1] if share_noise else torch.randn_like(u)
                u = u_mean + noise * delta
                if intermediate_save_indices is not None and i-1 in intermediate_save_indices:
                    intermediate_out.append((u, u_mean))

            return u_mean, config.model.K, [u_mean for (u, u_mean) in intermediate_out]

    return reconstructor


def get_reconstruction_fn_inverse_propagation_dpps(config, initial_measurement,
                                                   intermediate_save_indices, delta, device,
                                                   n_candidates=50, forward_module=None, share_noise=False):
    K = config.model.K

    def score_function(u_mean, noise_patch, patch_coord, config, initial_measurement):
        candidate = u_mean.clone()
        x0, y0, ph, pw = patch_coord
        candidate[:, :, x0:x0+ph, y0:y0+pw] += noise_patch * config.model.delta
        simulated_measurement = forward_module(candidate, K * torch.ones(initial_measurement.shape[0],
                                                     dtype=torch.long).to(config.device))
        return torch.mean((simulated_measurement - initial_measurement) ** 2)

    def generate_best_noise(u_mean, n_candidates_max, patch_size, i, config, initial_measurement):
        B, C, H, W = u_mean.shape
        ph, pw = patch_size
        best_noise = torch.zeros_like(u_mean)
        for x0 in range(0, H, ph):
            for y0 in range(0, W, pw):
                patch_coord = (x0, y0, ph, pw)
                best_score = torch.inf
                best_patch_noise = None
                n_candidates = n_candidates_max
                for _ in range(n_candidates):
                    noise_patch = torch.randn((B, C, ph, pw), device=u_mean.device)
                    score = score_function(u_mean, noise_patch, patch_coord, config, initial_measurement)
                    if score < best_score:
                        best_score = score
                        best_patch_noise = noise_patch
                best_noise[:, :, x0:x0+ph, y0:y0+pw] = best_patch_noise
        return best_noise

    def reconstructor(model):
        if share_noise:
            noises = [torch.randn_like(initial_measurement[0], dtype=torch.float)[None]
                      for _ in range(K)]

        intermediate_out = []
        with torch.no_grad():
            u = initial_measurement.to(config.device).float()
            if intermediate_save_indices is not None and K in intermediate_save_indices:
                intermediate_out.append((u, u))

            for i in range(K, 0, -1):
                vec_fwd_steps = torch.ones(
                    initial_measurement.shape[0], device=device, dtype=torch.long) * i
                u_mean = model(u, vec_fwd_steps) + u
                best_noise = generate_best_noise(u_mean=u_mean, n_candidates_max=n_candidates,
                                                 patch_size=(64, 64), i=i, config=config,
                                                 initial_measurement=initial_measurement)
                u = u_mean + best_noise * delta if i > 1 else u_mean
                if intermediate_save_indices is not None and i-1 in intermediate_save_indices:
                    intermediate_out.append((u, u_mean))

            return u_mean, config.model.K, [u_mean for (u, u_mean) in intermediate_out]

    return reconstructor


def safe_reconstruction(config, model_fn, initial_batch, device, share_noise=False):
    """Reconstruct each sample independently to keep memory usage bounded."""
    batch_size = initial_batch.shape[0]
    all_samples = []
    all_intermediates = []

    for idx in range(batch_size):
        single = initial_batch[idx:idx+1]
        recon_fn = get_reconstruction_fn_inverse_propagation(
            config,
            single,
            intermediate_save_indices=list(range(0, config.model.K+1, 1)),
            delta=config.model.delta,
            device=device,
            share_noise=share_noise,
        )
        with torch.no_grad():
            sample, _, intermediates = recon_fn(model_fn)
        all_samples.append(sample.cpu())
        all_intermediates.append([t.cpu() for t in intermediates])
        del single, recon_fn, sample, intermediates
        torch.cuda.empty_cache()

    final = torch.cat(all_samples, dim=0)
    merged_intermediates = [
        torch.cat([im[k] for im in all_intermediates], dim=0)
        for k in range(len(all_intermediates[0]))
    ]
    return final, merged_intermediates


def get_initial_measurement(config, forward_module, batch_size=None):
    """Fetch the initial measurement together with the GT and WF reference inputs."""
    _, testloader, WFloader = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        train_batch_size=batch_size,
    )

    gt_sample = next(iter(testloader))[0].to(config.device)
    wf_sample = next(iter(WFloader))[0].to(config.device)
    original_images = gt_sample.clone()
    initial_measurement = forward_module(
        gt_sample, config.model.K * torch.ones(gt_sample.shape[0], dtype=torch.long).to(config.device)
    )
    return initial_measurement, original_images, wf_sample
