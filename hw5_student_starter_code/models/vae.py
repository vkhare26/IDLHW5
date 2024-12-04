import torch
import torch.nn as nn
from contextlib import contextmanager

from .vae_modules import Encoder, Decoder
from .vae_distributions import DiagonalGaussianDistribution


class VAE(nn.Module):
    def __init__(self,
                 ### Encoder Decoder Related
                 double_z=True,
                 z_channels=3,
                 embed_dim=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
                 num_res_blocks=2):
        super(VAE, self).__init__()
        
        # Define encoder and decoder
        self.encoder = Encoder(
            in_channels=in_channels, ch=ch, out_ch=out_ch, 
            num_res_blocks=num_res_blocks, z_channels=z_channels, 
            ch_mult=ch_mult, resolution=resolution, double_z=double_z, 
            attn_resolutions=[]
        )
        self.decoder = Decoder(
            in_channels=in_channels, ch=ch, out_ch=out_ch, 
            num_res_blocks=num_res_blocks, z_channels=z_channels, 
            ch_mult=ch_mult, resolution=resolution, double_z=double_z, 
            attn_resolutions=[]
        )
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

    @torch.no_grad()
    def encode(self, x):
        """
        Encodes an input image into a sampled latent vector using the re-parameterization trick.
        
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
        
        Returns:
            posterior (torch.Tensor): Latent representation of shape (B, embed_dim, H/scale, W/scale).
        """
        # Pass through the encoder
        h = self.encoder(x)
        # Compute moments (mean and log variance) for Gaussian distribution
        moments = self.quant_conv(h)
        # Create a Gaussian distribution and sample from it
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample()

    @torch.no_grad()
    def decode(self, z):
        """
        Decodes a latent vector into a reconstructed image.
        
        Args:
            z (torch.Tensor): Latent representation of shape (B, embed_dim, H/scale, W/scale).
        
        Returns:
            dec (torch.Tensor): Reconstructed image of shape (B, C, H, W).
        """
        # Apply post-quantization convolution
        z = self.post_quant_conv(z)
        # Pass through the decoder to reconstruct the image
        dec = self.decoder(z)
        return dec

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        Initialize the model from a checkpoint.
        
        Args:
            path (str): Path to the checkpoint file.
            ignore_keys (list): List of keys to ignore when loading the state_dict.
        """
        # Load the state_dict from the checkpoint
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        # Remove ignored keys
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # Load the state_dict into the model
        keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(keys)
