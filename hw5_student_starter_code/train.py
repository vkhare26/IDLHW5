# Updated script with additional explanations for missing TODOs and better clarity
'''
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
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
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

class pipeline:
    @staticmethod
    def generate_images(unet, scheduler, num_samples, class_emb=None, device="cuda"):
        unet.eval()
        with torch.no_grad():
            samples = torch.randn((num_samples, 3, args.image_size, args.image_size), device=device)
            scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=device)
            for t in tqdm(scheduler.timesteps, desc="Sampling"):
                timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
                model_output = unet(samples, timesteps, class_emb)
                samples = scheduler.step(model_output, t, samples)
            samples = (samples + 1) / 2  # Rescale to [0, 1]
        return samples




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
    if is_primary(args):
      logger.info("***** Training arguments *****")
      logger.info(args)
      logger.info("***** Running training *****")
      logger.info(f"  Num examples = {len(train_dataset)}")
      logger.info(f"  Num Epochs = {args.num_epochs}")
      logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
      logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
      logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
      logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))


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

        # Validation loop
        unet.eval()  # Set model to evaluation mode
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)  # Set seed for reproducibility

        logger.info(f"Starting validation for epoch {epoch+1}/{args.num_epochs}...")

        # Generate a fixed number of images for validation
        num_validation_images = 16
        if args.use_cfg:
            # Randomly sample classes for conditional generation
            classes = torch.randint(0, args.num_classes, (num_validation_images,), device=device)
            class_emb = class_embedder(classes) if args.use_cfg else None
            gen_images = pipeline.generate_images(
                unet=unet,
                scheduler=scheduler,
                num_samples=num_validation_images,
                class_emb=class_emb,
                device=device
            )
        else:
            # Unconditional image generation
            gen_images = pipeline.generate_images(
                unet=unet,
                scheduler=scheduler,
                num_samples=num_validation_images,
                device=device
            )

        # Create an image grid for visualization
        grid_image = make_grid(gen_images, nrow=4, normalize=True, value_range=(0, 1))
        grid_image = transforms.ToPILImage()(grid_image)

        # Log generated images to WandB
        if args.use_wandb and is_primary(args):
            wandb.log({"generated_images": wandb.Image(grid_image, caption=f"Epoch {epoch+1}")})

        # Save validation images locally (optional)
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir, exist_ok=True)
        grid_image.save(os.path.join(output_dir, "validation_images.png"))

        logger.info(f"Validation for epoch {epoch+1}/{args.num_epochs} completed.")

            
        # save checkpoint
        if is_primary(args):
            save_checkpoint(uunet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp, class_embedder, optimizer, epoch, save_dir=save_dir)









    logger.info("Training completed.")

if __name__ == "__main__":
    main()
'''
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data 
    parser.add_argument("--data_dir", type=str, default='./data', help="data folder") 
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=16, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases for logging")
    # training
    parser.add_argument("--run_name", type=str, default=None, help="DDPM_Train_Latest_Run")
    parser.add_argument("--output_dir", type=str, default="/content/IDLHW5/hw5_student_starter_code/experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')
    
    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")
    
    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    
    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")
    
    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")
    
    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")
    
    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")
    
    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args

class Pipeline:
    @staticmethod
    def generate_images(unet, ddpm_scheduler, num_samples, args, class_emb=None, device="cuda"):
        unet.eval()
        with torch.no_grad():
            samples = torch.randn((num_samples, 3, args.image_size, args.image_size), device=device)
            ddpm_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=device)
            for t in tqdm(ddpm_scheduler.timesteps, desc="Sampling"):
                timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
                model_output = unet(samples, timesteps, class_emb)
                samples = ddpm_scheduler.step(model_output, t, samples)
            samples = (samples + 1) / 2  # Rescale to [0, 1]
        return samples

    
def main():
    
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    RESUME_LOGGING = True  # Set this to True if you want to resume a previous WandB run
    run_id = "pq1rgwhk"  # Specify the WandB run ID for resuming, if applicable


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
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
        
        # setup dataset
    logger.info("Creating dataset")

    # Transform to normalize CIFAR-10 images to [-1, 1] and apply horizontal flip
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize images to the specified size
        transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Use CIFAR-10 dataset for training
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Setup dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    shuffle = sampler is None  # Shuffle only if no distributed sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
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
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # DDPM scheduler
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range
    )

    

    # NOTE: this is for latent DDPM 
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt('/content/IDLHW5/hw5_student_starter_code/pretrained/model.ckpt')
        vae.eval()

    # NOTE: this is for Classifier-Free Guidance (CFG)
    class_embedder = None
    if args.use_cfg:
        # Instantiate ClassEmbedder with the number of classes + 1 for unconditional class
        class_embedder = ClassEmbedder(embed_dim=128, n_classes=args.num_classes + 1, cond_drop_rate=0.1)

            
    # send to device
    unet = unet.to(device)
    #lr_scheduler = lr_scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    # Setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False
        )
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False
            )
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae

    # Setup DDIM scheduler (if enabled)
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule
        )
    else:
        scheduler_wo_ddp = ddpm_scheduler

# Setup evaluation pipeline
# This pipeline is for evaluation purposes and is not differentiable
    pipeline = DDPMPipeline(
        unet=unet_wo_ddp,
        scheduler=scheduler_wo_ddp,
        vae=vae_wo_ddp,
        class_embedder=class_embedder_wo_ddp
    )

    
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(
            project='ddpm', 
            name=args.run_name, 
            config=vars(args))
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

  # Training loop
    for epoch in range(args.num_epochs):
      
      # Set epoch for distributed sampler, if applicable
      if hasattr(train_loader.sampler, 'set_epoch'):
          train_loader.sampler.set_epoch(epoch)

      args.epoch = epoch
      if is_primary(args):
          logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
      
      loss_m = AverageMeter()

      # Set unet and scheduler to train mode
      unet.train()
      

      for step, (images, labels) in enumerate(train_loader):
          batch_size = images.size(0)
          
          # Send images and labels to device
          images = images.to(device)
          labels = labels.to(device)
          
          # NOTE: This is for latent DDPM
          if vae is not None:
              # Use VAE to encode images as latents
              images = vae.encode(images).latent_dist.sample()
              # Ensure the latent has unit std
              images = images * 0.1845
          
          # Zero grad optimizer
          optimizer.zero_grad()
          
          # NOTE: This is for CFG
          if class_embedder is not None:
              # Use class embedder to get class embeddings
              class_emb = class_embedder(labels)
          else:
              # If not CFG, set class_emb to None
              class_emb = None
          
          # Sample noise
          noise = torch.randn_like(images)
          
          # Sample timestep t
          timesteps = torch.randint(0, args.num_train_timesteps, (batch_size,), device=device).long()
          
          # Add noise to images using scheduler
          noisy_images = ddpm_scheduler.add_noise(images, noise, timesteps)
          
          # Model prediction
          model_pred = unet(noisy_images, timesteps, class_emb)
          
          # Set the target based on the prediction type
          if args.prediction_type == 'epsilon':
              target = noise
          
          # Calculate loss
          loss = F.mse_loss(model_pred, target)
          
          # Record loss
          loss_m.update(loss.item())
          
          # Backward and step
          loss.backward()
          
          # Gradient clipping
          if args.grad_clip:
              torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
          
          # Step optimizer
          optimizer.step()
          lr_scheduler.step()
          
          # Update progress bar
          progress_bar.update(1)
          
          # Logging
          if step % 100 == 0 and is_primary(args):
              logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
              wandb_logger.log({'loss': loss_m.avg})

      # Validation
      # Send unet to evaluation mode
      unet.eval()        
      generator = torch.Generator(device=device)
      generator.manual_seed(epoch + args.seed)
      num_validation_images = 16 
      # NOTE: This is for CFG
      if args.use_cfg:
          # Randomly sample 4 classes
          classes = torch.randint(0, args.num_classes, (4,), device=device)
          # Generate images using pipeline with class embeddings
          gen_images = Pipeline.generate_images(
            unet=unet,
            ddpm_scheduler=ddpm_scheduler,
            num_samples=num_validation_images,
            args=args,
            class_emb=class_emb if args.use_cfg else None,
            device=device
)
      else:
          # Generate images using pipeline unconditionally
          gen_images = Pipeline.generate_images(
          unet=unet,
          ddpm_scheduler=ddpm_scheduler,
          num_samples=num_validation_images,
          args=args,
          class_emb=class_emb if args.use_cfg else None,
          device=device
)
              
      # Create a blank canvas for the grid
      # Ensure all generated images are PIL images
    grid_image = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))

    # Paste images into the grid
    for i, image in enumerate(gen_images):
        # Convert tensor to PIL image
        image = transforms.ToPILImage()(image)
        
        # Resize the image to the expected size
        image = image.resize((args.image_size, args.image_size), Image.ANTIALIAS)
        
        # Compute paste position
        x = (i % 4) * args.image_size
        y = 0
        
        # Paste the image onto the grid
        grid_image.paste(image, (x, y))

    # Log to WandB
        wandb.log({'generated_images': wandb.Image(grid_image, caption=f"Epoch {epoch+1}")})
              
        # Save checkpoint
        save_checkpoint(unet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp, class_embedder, optimizer, epoch, save_dir=save_dir)

    logger.info(f"Validation for epoch {epoch+1}/{args.num_epochs} completed.")


      
if __name__ == '__main__':
    main()


























