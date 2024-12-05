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
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)

# Define a function to load validation images
def load_validation_images(val_dir, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to a common size
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization
    ])
    
    validation_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    return validation_loader

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
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # Model setup
    logger.info("Creating model")
    unet = UNet(
        input_size=args.unet_in_size, input_ch=args.unet_in_ch,
        T=args.num_train_timesteps, ch=args.unet_ch,
        ch_mult=args.unet_ch_mult, attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout,
        conditional=args.use_cfg, c_dim=args.unet_ch
    )
    
    # Print the number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # Setup scheduler
    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(num_train_timesteps=args.num_train_timesteps).to(device)
    
    # VAE setup if using latent DDPM (for later use)
    vae = VAE().to(device) if args.latent_ddpm else None
    if vae:
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    
    # Class embedder setup if using classifier-free guidance
    class_embedder = ClassEmbedder(None).to(device) if args.use_cfg else None
    
    # Send U-Net to device
    unet = unet.to(device)
    
    # Load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # Setup pipeline
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)

    logger.info("***** Running Inference *****")
    
    # Generate images
    all_images = []
    if args.use_cfg:
        # Generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(batch_size=batch_size, class_labels=classes)
            all_images.append(gen_images)
    else:
        # Generate 5000 images
        for _ in tqdm(range(0, 5000, args.batch_size)):
            gen_images = pipeline(batch_size=args.batch_size)
            all_images.append(gen_images)
    
    # Load validation images as a reference batch
    val_dir = os.path.join(args.data_dir, 'validation')
    validation_loader = load_validation_images(val_dir=val_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Evaluate using torchmetrics
    import torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # Initialize FID and Inception Score metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)

    # Iterate over validation data and update metrics
    for val_images, _ in validation_loader:
        val_images = val_images.to(device)
        fid.update(val_images, real=True)

    # Update with generated images
    for gen_batch in all_images:
        fid.update(gen_batch, real=False)

    # Compute and log final FID and Inception scores
    fid_score = fid.compute().item()
    inception_score_value = inception_score.compute().item()
    
    logger.info(f"FID Score: {fid_score}")
    logger.info(f"Inception Score: {inception_score_value}")

if __name__ == '__main__':
    main()
