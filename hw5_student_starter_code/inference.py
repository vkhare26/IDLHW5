import os
import sys
import argparse
import numpy as np
from ruamel.yaml import YAML
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

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for a trained DDPM model.")
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="Path to config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output_dir", type=str, default='generated_images', help="Directory to save generated images.")
    return parser.parse_args()

def save_images(images, output_dir, batch_idx):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"generated_image_{batch_idx * len(images) + i}.png"))

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

    # Seed everything
    seed_everything(config['seed'])
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Model setup
    logger.info("Creating model")
    unet = UNet(
        input_size=config['unet_in_size'],
        input_ch=config['unet_in_ch'],
        T=config['num_train_timesteps'],
        ch=config['unet_ch'],
        ch_mult=config['unet_ch_mult'],
        attn=config['unet_attn'],
        num_res_blocks=config['unet_num_res_blocks'],
        dropout=config['unet_dropout'],
        conditional=config['use_cfg'],
        c_dim=config['unet_ch']
    ).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 10 ** 6:.2f}M parameters")

    # Scheduler
    if config['use_ddim']:
        scheduler = DDIMScheduler(
            num_train_timesteps=config['num_train_timesteps'],
            num_inference_steps=config['num_inference_steps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            beta_schedule=config['beta_schedule'],
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=config['num_train_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            beta_schedule=config['beta_schedule'],
        )
    scheduler.to(device)
    
    # VAE (Latent DDPM)
    vae = None
    if config.get('latent_ddpm', False):
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval().to(device)
    
    # Class Embedder (CFG)
    class_embedder = None
    if config.get('use_cfg', False):
        class_embedder = ClassEmbedder(num_classes=config['num_classes']).to(device)

    # Load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # Setup pipeline
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder)

    logger.info("***** Running Inference *****")

    # Generate images and save them
    num_images = 5000
    batch_size = config['batch_size']
    generator = torch.Generator(device=device)
    generator.manual_seed(config['seed'])

    batch_idx = 0
    if config.get('use_cfg', False):
        # Generate 50 images per class if using CFG
        for i in range(config['num_classes']):
            logger.info(f"Generating 50 images for class {i}")
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(batch_size=batch_size, classes=classes, generator=generator)
            save_images(gen_images, args.output_dir, batch_idx)
            batch_idx += 1
    else:
        # Generate 5000 images without classes
        for _ in tqdm(range(0, num_images, batch_size)):
            gen_images = pipeline(batch_size=batch_size, generator=generator)
            save_images(gen_images, args.output_dir, batch_idx)
            batch_idx += 1

    logger.info(f"Images saved in {args.output_dir}")

if __name__ == '__main__':
    main()
