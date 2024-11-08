from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch 
import torch.nn as nn
from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler
        
        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae
            
        # NOTE: this is for CFG
        self.class_embedder = class_embedder if class_embedder is not None else None

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale : Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device
        
        # NOTE: this is for CFG
        if classes is not None or guidance_scale is not None:
            assert self.class_embedder is not None, "class_embedder is not defined"
        
        if classes is not None:
            # convert classes to tensor
            if isinstance(classes, int):
                classes = [classes] * batch_size
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "Length of classes must be equal to batch_size"
            classes = torch.tensor(classes, device=device)
            
            # Get unconditional and conditional class embeddings
            uncond_classes = torch.zeros_like(classes)  # Zero as a placeholder for "unconditional" class
            class_embeds = self.class_embedder(classes)
            uncond_embeds = self.class_embedder(uncond_classes)
        
        # Start with random noise
        image = torch.randn(image_shape, generator=generator, device=device)

        # Set step values using scheduler's set_timesteps function
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Inverse diffusion process
        for t in self.progress_bar(self.scheduler.timesteps):
            # Classifier-Free Guidance (CFG) setup
            if guidance_scale is not None and guidance_scale != 1.0:
                # Duplicate model inputs for conditional/unconditional guidance
                model_input = torch.cat([image, image], dim=0)
                c = torch.cat([uncond_embeds, class_embeds], dim=0)
            else:
                model_input = image
                c = None  # Leave as None if CFG is not used
            
            # 1. Predict noise model output
            model_output = self.unet(model_input, t, class_emb=c)
            
            if guidance_scale is not None and guidance_scale != 1.0:
                # Separate the outputs for CFG
                uncond_model_output, cond_model_output = model_output.chunk(2)
                model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)
            
            # 2. Compute the previous image in the reverse diffusion process
            image = self.scheduler.step(model_output, t, image)["prev_sample"]
        
        # For latent DDPM with VAE
        if self.vae is not None:
            image = self.vae.decode(image / 0.18215)  # Scale as needed
            image = image.clamp(-1, 1)  # Clamp to the valid range
        
        # Rescale to [0, 1] for final output
        image = (image + 1) / 2
        
        # Convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
