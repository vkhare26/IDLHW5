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
from torchvision.utils import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # Config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="config file used to specify parameters")
    
    # Other arguments needed in the script
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/test/IDLHW5/output", help="Directory for output files")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the experiment run")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")

    # Model-specific arguments
    parser.add_argument("--image_size", type=int, default=32, help="Input image size")
    parser.add_argument("--unet_in_size", type=int, default=32, help="Input size for UNet")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="Number of input channels for UNet")
    parser.add_argument("--unet_ch", type=int, default=64, help="Base number of channels for UNet")
    parser.add_argument("--unet_ch_mult", type=list, default=[1, 2, 4, 8], help="Channel multiplier for each layer of UNet")
    parser.add_argument("--unet_attn", type=bool, default=False, help="Use attention in UNet")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="Number of residual blocks in UNet")
    parser.add_argument("--unet_dropout", type=float, default=0.1, help="Dropout rate in UNet")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of timesteps for training")
    parser.add_argument("--use_cfg", type=bool, default=False, help="Whether to use classifier-free guidance")

    # Distributed training arguments
    parser.add_argument("--distributed", type=str2bool, default=False, help="Enable distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed training")
    parser.add_argument("--rank", type=int, default=0, help="Rank for distributed training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    # First parse command-line args to check for config file
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
    
    # WandB authentication setup
    wandb.login(key="f5e8f40b42dcfd326f8e06e3cc6bae069adeb506")  # Uncomment this line to use direct login with the API key

    # Initialize wandb
    wandb.init(
        project="HWP5",  # Replace with your project name
        name=args.run_name,
        config=args  # Logs all args as configuration
    )
    
    # Seed everything
    seed_everything(args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    # Setup dataset
    logger.info("Creating CIFAR-10 dataset")
    # Transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize to input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # CIFAR-10 normalization
    ])
    
    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Calculate total batch_size
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
    # UNet and scheduler initialization
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)  # Example scheduler initialization
    vae = VAE()  # Initialize VAE if needed

    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Send models to device
    unet = unet.to(device)
    vae = vae.to(device)
    
    # Training loop
    logger.info("Training begins...")
    for epoch in range(args.num_epochs):
        loss_m = AverageMeter()
        for step, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            
            # TODO: Add training logic here, compute loss, update optimizer
            # e.g., optimizer.zero_grad(), loss.backward(), optimizer.step()

            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{len(train_loader)}, Loss {loss_m.avg}")
                wandb.log({'loss': loss_m.avg})
                
            if step % 500 == 0 and is_primary(args):
                save_checkpoint(unet, scheduler, vae, optimizer=optimizer, epoch=epoch, save_dir=save_dir)
    
    # Final save checkpoint
    save_checkpoint(unet, scheduler, vae, optimizer=optimizer, epoch=epoch, save_dir=save_dir)
    
if __name__ == "__main__":
    main()
