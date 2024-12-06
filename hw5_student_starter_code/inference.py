import os 
import logging 
import torch
from torchvision import datasets, transforms
from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint
from train import parse_args
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.datasets import CIFAR10


logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

def show_images(images, title="Generated Images", n_cols=8):
    """
    Display a batch of images in a grid.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        title (str): Title of the plot.
        n_cols (int): Number of columns in the grid.
    """
    images = images.permute(0, 2, 3, 1)  # Convert from (B, C, H, W) to (B, H, W, C)
    images = images.clamp(0, 1)  # Ensure values are in [0, 1] for visualization

    n_rows = (len(images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img.cpu().numpy())
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

RESUME_LOGGING = False  # Set this to True if you want to resume a previous WandB run
run_id = "pq1rgwhk"  # Specify the WandB run ID for resuming, if applicable

'''
def load_validation_images(validation_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(validation_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
'''

def load_cifar10_images(image_size, batch_size, train=True):
    """
    Load CIFAR-10 dataset and return a DataLoader.
    
    Args:
        image_size (int): Desired image size (CIFAR-10 is 32x32 by default, resizing is optional).
        batch_size (int): Batch size for DataLoader.
        train (bool): Whether to load the training set or the test set.

    Returns:
        DataLoader: PyTorch DataLoader for CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def main():
    args = parse_args()
    seed_everything(args.seed)

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
    ).to(device)
    logger.info(f"Number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")

    scheduler_class = DDIMScheduler if args.use_ddim else DDPMScheduler
    scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)

    vae = VAE().to(device) if args.latent_ddpm else None
    if vae:
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    class_embedder = ClassEmbedder(128, args.num_classes + 1, cond_drop_rate=0.1).to(device) if args.use_cfg else None

    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)

    pipeline = DDPMPipeline(
        unet=unet, scheduler=scheduler,
        vae=vae, class_embedder=class_embedder
    )

    logger.info("***** Running Inference *****")
    all_images = []
    batch_size = 50 if args.use_cfg else 64

    if args.use_cfg:
      for i in tqdm(range(args.num_classes), desc="Generating images per class"):
          classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
          gen_images = pipeline(batch_size=batch_size, class_emb=class_embedder(classes), device=device)
          if isinstance(gen_images, list):  # If pipeline returns a list of tensors
              gen_images = torch.stack(gen_images)  # Convert list of tensors to a single tensor
          all_images.append(gen_images.cpu())
          show_images(gen_images, title=f"Generated Images - Batch {idx}")
    else:
      for _ in tqdm(range(0, 5000, batch_size), desc="Generating 5000 images"):
          gen_images = pipeline(batch_size=batch_size, device=device)
          if isinstance(gen_images, list):  # If pipeline returns a list of tensors
              gen_images = torch.stack(gen_images)  # Convert list of tensors to a single tensor
          all_images.append(gen_images.cpu())
          show_images(gen_images, title=f"Generated Images - Batch {idx}")




    #validation_loader = load_validation_images(args.validation_dir, args.image_size, batch_size=args.batch_size)
    validation_loader = load_cifar10_images(image_size=args.image_size, batch_size=args.batch_size, train=False)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)

    for ref_images, _ in tqdm(validation_loader, desc="Updating FID with validation images"):
        fid.update(ref_images.to(device), real=True)

    for gen_batch in all_images:
        for gen_image in gen_batch:
            fid.update(gen_image.unsqueeze(0).to(device), real=False)
            inception_score.update(gen_image.unsqueeze(0).to(device))

    final_fid = fid.compute().item()
    final_is, _ = inception_score.compute()
    logger.info(f"FID: {final_fid}")
    logger.info(f"Inception Score: {final_is}")

if __name__ == "__main__":
    main()
