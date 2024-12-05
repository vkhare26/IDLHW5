# Updated script with additional explanations for missing TODOs and better clarity

import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint


logger = get_logger(__name__)
RESUME_LOGGING = False  # Set this to True if you want to resume a previous WandB run
run_id = ""  # Specify the WandB run ID for resuming, if applicable


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Config file used to specify parameters")

    # Data
    parser.add_argument("--data_dir", type=str, default='./data', help="Data folder") 
    parser.add_argument("--image_size", type=int, default=128, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloading")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes in dataset")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases for logging")



    # Training
    parser.add_argument("--run_name", type=str, default="DDPM_scheduler_first_run", help="Run name for the experiment")
    parser.add_argument("--output_dir", type=str, default="/content/IDLHW5/hw5_student_starter_code/experiments", help="Output folder")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help="Mixed precision type")

    # DDPM
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="DDPM training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="DDPM inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="DDPM beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="DDPM beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="DDPM beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="DDPM variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="DDPM prediction type (e.g., epsilon)")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="Whether to clip sample at each reverse step")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="Clip sample range")

    # UNet
    parser.add_argument("--unet_in_size", type=int, default=128, help="UNet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="UNet input channels")
    parser.add_argument("--unet_ch", type=int, default=128, help="UNet base channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="UNet channel multipliers")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="UNet attention stages")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="Number of residual blocks in UNet")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="UNet dropout probability")

    # VAE
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="Use VQVAE for latent DDPM")

    # CFG
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="Use classifier-free guidance (CFG) for conditional DDPM")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="CFG scale for inference")

    # DDIM sampler
    parser.add_argument("--use_ddim", type=str2bool, default=True, help="Use DDIM sampler for inference")

    # Checkpoint
    parser.add_argument("--ckpt", type=str, default="/content/IDLHW5/hw5_student_starter_code/checkpoints", help="Checkpoint path for inference")

    # First parse to check for config file
    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # Re-parse to allow CLI args to overwrite config file defaults
    args = parser.parse_args()
    return args

def sample_images(unet, scheduler, num_samples, image_size, device):
    """
    Generate samples from the diffusion model.

    Args:
        unet: The trained UNet model.
        scheduler: The scheduler used for sampling (DDPMScheduler).
        num_samples: Number of images to generate.
        image_size: Size of the output images.
        device: The device to run inference on.

    Returns:
        Tensor of generated images.
    """
    unet.eval()
    with torch.no_grad():
        samples = torch.randn((num_samples, 3, image_size, image_size), device=device)
        scheduler.set_timesteps(num_inference_steps=1000, device=device)

        for t in tqdm(scheduler.timesteps, desc="Sampling"):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            model_output = unet(samples, timesteps)
            samples = scheduler.step(model_output, t, samples)

        samples = (samples + 1) / 2
        return samples.clamp(0, 1).cpu()



def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.use_wandb:
        if RESUME_LOGGING and run_id:
            run = wandb.init(
                id=run_id,
                resume="allow",
                project="HWP5",
                config=vars(args)
            )
        else:
            run = wandb.init(
                name=args.run_name,
                reinit=True,
                project="HWP5",
                config=vars(args)
            )

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Device setup
    device = init_distributed_device(args)
    logger.info(f"Training on device: {device}")

    # Load CIFAR-10 dataset
    logger.info("Loading CIFAR-10 dataset")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Dataset loaded with {len(train_dataset)} samples.")

    # Model setup
    logger.info("Creating model")
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=128,  # Match ClassEmbedder embed_dim
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)

    if args.use_cfg:
        class_embedder = ClassEmbedder(embed_dim=128, n_classes=args.num_classes + 1).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Model created with {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M parameters.")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        unet.train()
        loss_meter = AverageMeter()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generate noisy images and predictions
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, args.num_train_timesteps, (images.size(0),), device=device).long()
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            class_emb = class_embedder(labels) if args.use_cfg else None
            model_pred = unet(noisy_images, timesteps, class_emb)

            # Loss and optimization
            loss = F.mse_loss(model_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            optimizer.step()
            loss_meter.update(loss.item())

            # Log progress
            if step % 100 == 0:
                logger.info(f"Epoch {epoch}/{args.num_epochs}, Step {step}, Loss: {loss_meter.avg:.4f}")
                if args.use_wandb:
                    wandb.log({"epoch": epoch, "step_loss": loss_meter.avg})

    logger.info("Training completed.")

if __name__ == "__main__":
    main()
