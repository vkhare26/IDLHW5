import os
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import UNet
from schedulers import DDPMScheduler
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool

logger = logging.getLogger(__name__)

RESUME_LOGGING = False  # Set this to True if you want to resume a previous WandB run
run_id = ""  # Specify the WandB run ID for resuming, if applicable


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Config file specifying parameters")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory for CIFAR-10 dataset")
    parser.add_argument("--image_size", type=int, default=32, help="Image size (CIFAR-10 is 32x32)")
    parser.add_argument("--batch_size", type=int, default=64, help="Per GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in CIFAR-10")

    # Training parameters
    parser.add_argument("--run_name", type=str, default="ddpm_experiment", help="Run name for logging")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output folder")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--distributed", type=str2bool, default=False, help="Enable distributed training")

    # DDPM parameters
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="DDPM training timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="DDPM beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="DDPM beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="DDPM beta schedule")

    # UNet parameters
    parser.add_argument("--unet_in_size", type=int, default=32, help="UNet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="UNet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="UNet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 4], nargs='+', help="UNet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1], nargs='+', help="UNet attention stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="UNet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="UNet dropout")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            yaml_data = yaml.safe_load(f)
            parser.set_defaults(**yaml_data)

    return parser.parse_args()


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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize device and distributed settings
    device = init_distributed_device(args)
    logger.info(f'Training on device: {device}')

    # WandB setup
    if args.use_wandb:
        wandb.login(key="fff82b95e4aff9cd6a6e538a6ffef767ab8865f5", relogin=True)
        if RESUME_LOGGING:
            run = wandb.init(
                id=run_id,
                resume=True,
                project="HWP5",
            )
        else:
            run = wandb.init(
                name=args.run_name,
                reinit=True,
                project="HWP5",
                config=vars(args),
            )

    # Setup CIFAR-10 dataset with normalization
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=args.num_workers, sampler=sampler)

    # Initialize model, scheduler, and optimizer
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=False,  # Adjust based on whether the model is conditional
        c_dim=args.unet_ch
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    logger.info("Training begins...")
    for epoch in range(args.num_epochs):
        loss_m = AverageMeter()
        for step, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)

            # Sample random timesteps and noise
            timesteps = torch.randint(0, args.num_train_timesteps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)

            # Add noise
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Model prediction and loss calculation
            model_output = unet(noisy_images, timesteps)
            loss = F.mse_loss(model_output, noise)
            loss_m.update(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            optimizer.step()

            if step % 100 == 0:
                wandb.log({
                    "step_loss": loss.item(),
                    "epoch": epoch,
                    "grad_norm": grad_norm,
                    "current_timestep": timesteps.float().mean().item(),  # Convert timesteps to float
                    "gpu_memory_allocated": torch.cuda.memory_allocated(device),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(device)
                })

        # Log epoch loss
        wandb.log({"epoch_loss": loss_m.avg, "epoch": epoch})

        # Generate and log images after every epoch
        generated_images = sample_images(
            unet=unet,
            scheduler=scheduler,
            num_samples=16,
            image_size=args.image_size,
            device=device
        )
        wandb.log({"generated_images": [wandb.Image(img.permute(1, 2, 0).numpy()) for img in generated_images]})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
