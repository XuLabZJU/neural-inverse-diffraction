# predict.py

import os
from pathlib import Path
import logging
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm

from scripts import reconstruction
from scripts import datasets
from scripts import utils
from model_code import utils as mutils

# ---------------- flags ----------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("checkpoint", 0, "Checkpoint number to use for custom sampling")
flags.DEFINE_string("data_dir", None, "Directory with test images (contains WF/LR images)")
flags.DEFINE_bool("direct", True, "Whether to directly use the image as initial sample (True for real WF)")
flags.DEFINE_integer("batch_size", 1, "Batch size (use 1 for deterministic per-image processing)")
flags.mark_flags_as_required(["config", "workdir", "checkpoint", "data_dir"])

flags.DEFINE_bool("save_process", False, "Whether to save GIF/MP4 of intermediate reconstructions")
flags.DEFINE_integer("n_candidates", 1, "Number of candidate reconstructions to keep")


# ---------------- main  ----------------
def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir
    checkpoint = FLAGS.checkpoint
    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size
    delta = config.model.delta
    device = config.device
    save_process = FLAGS.save_process
    n_candidates = FLAGS.n_candidates

    # Load the model using the same checkpoint logic as the original script.
    if checkpoint > 0:
        ckpt_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(config, ckpt_dir, checkpoint)
    else:
        ckpt_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, ckpt_dir)
    model_fn = mutils.get_model_fn(model, train=False)

    # Build the forward propagation module.
    scales = config.model.z_schedule
    forward_module = mutils.create_propagator_from_z_schedule(config, scales, device)

    # Preserve the original reconstruct.py output structure.
    val_dir = os.path.join(workdir, "additional_reconstructions")

    # Load data deterministically without augmentation.
    loader, file_list = datasets.load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=config.data.image_size,
        deterministic=True,
        random_flip=False,
        random_enhance=False,
        drop_last=False,
        return_paths=True,
        shard=0,
        num_shards=1,
    )

    logging.info(f"Deterministic sampling on {len(file_list)} images, saving to {val_dir}")

    # Track progress by image count.
    pbar = tqdm(total=len(file_list), unit="img", desc="Sampling", leave=True)

    # Process images one by one; batch_size=1 is recommended.
    for idx, batch in enumerate(loader):
        images, kwargs = batch
        if not torch.is_tensor(images):
            images = torch.tensor(images)
        cur_batch_size = images.shape[0]

        for b in range(cur_batch_size):
            file_idx = idx * batch_size + b
            if file_idx >= len(file_list):
                continue
            input_path = file_list[file_idx]
            fname = os.path.basename(input_path)

            initial_sample = images[b:b+1].to(device)

            # Use the multi-candidate reconstruction function.
            sampling_fn = reconstruction.get_reconstruction_fn_inverse_propagation_dpps(
                config, initial_sample,
                intermediate_save_indices=list(range(0, config.model.K+1, 1)),
                delta=delta,
                device=device,
                n_candidates=n_candidates, # 1 for w/o selection and measurement consistency
                forward_module=forward_module
            )

            # Run the reconstruction
            with torch.no_grad():
                sample, _, intermediates = sampling_fn(model_fn)

            # Derive the output filename and save the result.
            fname_base, _ = os.path.splitext(fname)
            out_name = fname_base + ".png"
            out_dir = os.path.join(val_dir, f"checkpoint_{checkpoint}")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(out_dir, out_name)

            # Save the output image
            utils.save_gray(out_dir, sample[0].detach().cpu(), out_name)

            # Optionally save intermediate process as GIF/MP4.
            if save_process and intermediates is not None:
                try:
                    # intermediates is expected to be a list/tensor sequence suitable for utils.save_gif
                    save_dir = out_dir
                    utils.save_gif(save_dir, intermediates, name=f"{os.path.splitext(out_name)[0]}_recon_process.gif")
                    utils.save_video(save_dir, intermediates, name=f"{os.path.splitext(out_name)[0]}_recon_process.mp4")
                except Exception as e:
                    logging.warning(f"Failed to save process video/gif for {out_name}: {e}")

            print(f"[{file_idx+1}/{len(file_list)}] {input_path} -> {out_path}")

            pbar.update(1)

    pbar.close()

    print("All done. Results saved in:", val_dir)


if __name__ == "__main__":
    app.run(main)

