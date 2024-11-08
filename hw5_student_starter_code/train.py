import os
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
from torchvision.utils import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # Config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="Config file specifying parameters")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default='/Users/vkhare26/Documents/IDL/HW5/imagenet100_128x128/', help="Data folder")
    parser.add_argument("--image_size", type=int, default=128, help="Image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Per GPU batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes in dataset")

    # Training parameters
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output folder")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='Mixed precision setting')
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Use Weights & Biases for logging")
    
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
    parser.add_argument("--unet_in_size", type=int, default=128, help="UNet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="UNet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="UNet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="UNet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="UNet attention stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="UNet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="UNet dropout")

    # Optional parameters
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="Use VAE for latent DDPM")
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="Use CFG for conditional DDPM")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="Use DDIM sampler for inference")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for inference")
    
    # First parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # Re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args
    
    
def main():
    
    # Parse arguments
    args = parse_args()
    
    # Seed everything
    seed_everything(args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize distributed training and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    # Setup dataset and data loader
    logger.info("Creating dataset")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    # Calculate total batch size
    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size
    
    # Setup experiment folder
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    
    # Setup model
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
        c_dim=args.unet_ch
    )
    
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # DDPM Scheduler setup
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    
    # Optional components for Latent DDPM and CFG
    vae = VAE().to(device) if args.latent_ddpm else None
    class_embedder = ClassEmbedder().to(device) if args.use_cfg else None
    
    # Send model and scheduler to device
    unet = unet.to(device)
    if vae:
        vae.eval()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Setup wandb logging if needed
    if args.rank == 0 and args.use_wandb:
        wandb.init(project="diffusion", config=args)
    
    # Training loop
    logger.info("Training begins...")
    for epoch in range(args.num_epochs):
        loss_m = AverageMeter()
        for step, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            
            # Sample random timesteps and noise
            batch_size = images.size(0)
            timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device).long()
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
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{len(train_loader)}, Loss {loss.item()} ({loss_m.avg})")
                if args.use_wandb:
                    wandb.log({'loss': loss_m.avg})
                
            if step % 500 == 0 and is_primary(args):
                save_checkpoint(unet, optimizer, epoch, loss_m.avg, save_dir)
    
    # Final save checkpoint
    save_checkpoint(unet, optimizer, epoch, loss_m.avg, save_dir)
    
if __name__ == "__main__":
    main()
