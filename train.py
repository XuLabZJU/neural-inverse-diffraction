import os
from pathlib import Path
import logging
from scripts import losses
from scripts import reconstruction
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
    train(FLAGS.config, FLAGS.workdir)


def train(config, workdir):
    """Runs the training pipeline. 
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """

    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")

    # Create directory for saving reconstructions
    recon_dir = os.path.join(workdir, "reconstructions")
    Path(recon_dir).mkdir(parents=True, exist_ok=True)
    # Create directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    model_evaluation_fn = mutils.get_model_fn(model, train=False)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(
        workdir, "checkpoints-meta", "checkpoint.pth")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(checkpoint_meta_dir)).mkdir(
        parents=True, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    trainloader, testloader, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)
    
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the optical forward propagator
    z_schedule = config.model.z_schedule
    forward_propagator = mutils.create_propagator_from_z_schedule(
        config, z_schedule, config.device)

    # Get the loss function
    train_step_fn = losses.get_step_fn(train=True, config=config, optimize_fn=optimize_fn,
                                    forward_propagator=forward_propagator)
    eval_step_fn = losses.get_step_fn(train=False, config=config, optimize_fn=optimize_fn,
                                    forward_propagator=forward_propagator)

    # Building sampling functions
    initial_sample, original_images, wf_images = reconstruction.get_initial_measurement(config, forward_propagator)
    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))

    for step in range(initial_step, num_train_steps + 1):
        print("step:", step)
        try:
            batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            batch = next(train_iter)[0].to(config.device).float()
        loss, losses_batch, fwd_steps_batch = train_step_fn(state, batch)

        writer.add_scalar("training_loss", loss.item(), step)
        
        # Report the loss on an evaluation dataset periodically
        if step != 0 and step % config.training.eval_freq == 0:
            logging.info("Starting evaluation")
            # Use 25 batches for test-set evaluation, arbitrary choice
            N_evals = 25
            for i in range(N_evals):
                try:
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                eval_loss, losses_batch, fwd_steps_batch = eval_step_fn(state, eval_batch)
                eval_loss = eval_loss.detach()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info("Saving a checkpoint")
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)

        # Generate samples periodically
        if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps:
            logging.info("Reconstructing...")
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            
            # Reconstruct samples one image at a time to avoid batch coupling
            sample, intermediate_samples = reconstruction.safe_reconstruction(config=config, model_fn=model_evaluation_fn,
                initial_batch=initial_sample, device=config.device)
            
            ema.restore(model.parameters())
            this_sample_dir = os.path.join(recon_dir, "iter_{}".format(step))
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "final.np")
            utils.save_gray(this_sample_dir, sample, "final.png")
            if initial_sample != None:
                utils.save_gray(this_sample_dir, initial_sample, "init.png")
            utils.save_gray(this_sample_dir, original_images, "original.png")

            utils.save_gif(this_sample_dir, intermediate_samples)
            utils.save_video(this_sample_dir, intermediate_samples)
        
        # Run WF reconstruction on the evaluation cadence
        if step != 0 and step % config.training.eval_freq == 0:
            logging.info("Running direct WF reconstruction at step %d" % step)
            
            # Execute the reconstruction pass
            sample, intermediate_samples = reconstruction.safe_reconstruction(config=config, model_fn=model_evaluation_fn,
                initial_batch=wf_images, device=config.device)
            
            # Persist reconstruction artifacts for inspection
            this_sample_dir = os.path.join(recon_dir, f"wf_recon_iter_{step}")
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "wf_final.np")
            utils.save_gray(this_sample_dir, sample, "wf_final.png")
            utils.save_gray(this_sample_dir, wf_images, "wf_input.png")
            utils.save_gif(this_sample_dir, intermediate_samples, name="wf_process.gif")
            utils.save_video(this_sample_dir, intermediate_samples, name="wf_process.mp4")


if __name__ == "__main__":
    app.run(main)

