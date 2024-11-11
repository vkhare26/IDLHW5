import os
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")

    # Config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="Config file specifying parameters")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory for CIFAR-10 dataset")
    parser.add_argument("--image_size", type=int, default=32, help="Image size (CIFAR-10 is 32x32)")
    parser.add_argument("--batch_size", type=int, default=64, help="Per GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in CIFAR-10")

    # Training parameters
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/test/IDLHW5/output", help="Output folder")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='Mixed precision setting')
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Use Weights & Biases for logging")
    parser.add_argument("--distributed", type=str2bool, default=False, help="Enable distributed training")

    # DDPM parameters
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="DDPM training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="DDPM inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="DDPM beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="DDPM beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="DDPM beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="DDPM variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="DDPM prediction type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="Whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="Clip sample range")

    # UNet parameters
    parser.add_argument("--unet_in_size", type=int, default=32, help="UNet input image size (CIFAR-10 is 32x32)")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="UNet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="UNet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 4], nargs='+', help="UNet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1], nargs='+', help="UNet attention stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="UNet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="UNet dropout")

    # Optional parameters
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="Use VAE for latent DDPM")
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="Use CFG for conditional DDPM")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="Use DDIM sampler for inference")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for inference")

    args = parser.parse_args()

    # Load config file if specified
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # Re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize device and distributed settings
    device = init_distributed_device(args) 
    logger.info(f'Training on device: {device}')

    # Setup CIFAR-10 dataset with correct normalization
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])  
    ])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=args.num_workers, sampler=sampler
    )

    # Initialize model, scheduler, and optimizer
    unet = UNet(
        input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps,
        ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout,
        conditional=args.use_cfg, c_dim=args.unet_ch
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule, variance_type=args.variance_type,
        prediction_type=args.prediction_type, clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )

    vae = VAE().to(device) if args.latent_ddpm else None
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize wandb logging if needed
    if args.use_wandb:
        wandb.init(project="diffusion", config=args)

    # Training loop
    logger.info("Training begins...")
    for epoch in range(args.num_epochs):
        loss_m = AverageMeter()
        for step, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            
            # Sample random timesteps and noise
            timesteps = torch.randint(0, args.num_train_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)
            
            # Add noise to images
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            
            # Model prediction and loss calculation
            model_output = unet(noisy_images, timesteps)
            loss = F.mse_loss(model_output, noise)
            loss_m.update(loss.item())
            
            # Backward pass and optimizer update
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            optimizer.step()
            
            # Logging
            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}, Loss: {loss.item()} ({loss_m.avg})")
                if args.use_wandb:
                    wandb.log({'loss': loss_m.avg})
                
            # Checkpoint saving
            if step % 500 == 0 and is_primary(args):
                if vae is not None:
                    save_checkpoint(unet, scheduler, vae=vae, optimizer=optimizer, epoch=epoch, save_dir=args.output_dir)
                else:
                    save_checkpoint(unet, scheduler, optimizer=optimizer, epoch=epoch, save_dir=args.output_dir)

    # Final checkpoint save
    if vae is not None:
        save_checkpoint(unet, scheduler, vae=vae, optimizer=optimizer, epoch=epoch, save_dir=args.output_dir)
    else:
        save_checkpoint(unet, scheduler, optimizer=optimizer, epoch=epoch, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
