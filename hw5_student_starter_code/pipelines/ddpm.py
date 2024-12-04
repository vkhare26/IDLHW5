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
        
        # Support for latent DDPM
        self.vae = vae if vae is not None else None

        # Support for Classifier-Free Guidance (CFG)
        self.class_embedder = class_embedder if class_embedder is not None else None

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to PIL images.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # Special case for grayscale (single-channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        """
        Create a progress bar for visual feedback.
        """
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
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        """
        Perform the reverse diffusion process to generate images.
        """
        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device
        
        # CFG validation
        if classes is not None or guidance_scale is not None:
            assert self.class_embedder is not None, "Class embedder is required for conditional generation."

        # Handle class embeddings if classes are specified
        if classes is not None:
            # Convert classes to tensor
            if isinstance(classes, int):
                classes = [classes] * batch_size
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "Length of classes must match batch size."
            classes = torch.tensor(classes, device=device)

            # Compute unconditional and conditional embeddings
            uncond_classes = torch.zeros_like(classes)  # Placeholder for unconditional class
            class_embeds = self.class_embedder(classes)
            uncond_embeds = self.class_embedder(uncond_classes)
        else:
            class_embeds = uncond_embeds = None

        # Start with random noise
        image = randn_tensor(image_shape, generator=generator, device=device)

        # Initialize the scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Reverse diffusion process
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale is not None and guidance_scale != 1.0:
                # CFG: Duplicate inputs for conditional/unconditional guidance
                model_input = torch.cat([image, image], dim=0)
                c = torch.cat([uncond_embeds, class_embeds], dim=0) if class_embeds is not None else None
            else:
                model_input = image
                c = class_embeds

            # Predict noise using the model
            model_output = self.unet(model_input, t, class_emb=c)

            if guidance_scale is not None and guidance_scale != 1.0:
                # CFG: Combine conditional and unconditional outputs
                uncond_model_output, cond_model_output = model_output.chunk(2)
                model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)

            # Compute the next step
            image = self.scheduler.step(model_output, t, image)["prev_sample"]
        
        # Decode latent variables if using VAE
        if self.vae is not None:
            image = self.vae.decode(image / 0.18215)  # Scaling factor depends on training setup
            image = image.clamp(-1, 1)  # Ensure values are in valid range
        
        # Rescale to [0, 1] for final output
        image = (image + 1) / 2
        
        # Convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
