'''
import torch
import os

def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, checkpoint_path='checkpoints/checkpoint.pth'):
    
    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    
    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("loading scheduler")
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])
    
    
        

def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
'''
import torch
import os

def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, checkpoint_path='checkpoints/checkpoint.pth', device='cpu'):
    """Load model and optimizer states from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load UNet and scheduler states
    print("Loading UNet state")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("Loading scheduler state")
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Optionally load VAE and class embedder states if available
    if vae is not None and 'vae_state_dict' in checkpoint:
        print("Loading VAE state")
        vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("Loading Class Embedder state")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])
    
    # Optionally load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        print("Loading optimizer state")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Optionally load epoch if available
    if 'epoch' in checkpoint:
        print(f"Resuming from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    else:
        return None


def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, save_dir='checkpoints'):
    """Save model and optimizer states to a checkpoint."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

    # Prepare the checkpoint dictionary
    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history to keep only the latest `keep_last_n` checkpoints
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    """Remove older checkpoints, keeping only the latest `keep_last_n` checkpoints."""
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n:
        for checkpoint_file in checkpoints[:-keep_last_n]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
