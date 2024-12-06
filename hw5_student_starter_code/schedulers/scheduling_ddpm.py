from typing import List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import randn_tensor


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        variance_type: str = "fixed_small",
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super(DDPMScheduler, self).__init__()

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Calculate betas
        if self.beta_schedule == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Beta schedule {self.beta_schedule} not implemented.")
        self.register_buffer("betas", betas)

        # Calculate alphas and their cumulative products
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Precompute timesteps
        timesteps = torch.arange(self.num_train_timesteps, dtype=torch.long)
        self.register_buffer("timesteps", timesteps)

    def set_timesteps(self, num_inference_steps: int = 250, device: Union[str, torch.device] = None):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps: {num_inference_steps} cannot be larger than self.num_train_timesteps:"
                f" {self.num_train_timesteps}."
            )
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=int)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator=None):
      t = timestep
      prev_t = t - 1 if t > 0 else None

      alpha_prod_t = self.alphas_cumprod[t]
      alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t is not None else torch.tensor(1.0, device=alpha_prod_t.device)

      if self.prediction_type == 'epsilon':
          pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
      else:
          raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

      if self.clip_sample:
          pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

      pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev)
      current_sample_coeff = torch.sqrt(1.0 - alpha_prod_t_prev)

      pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output

      if t > 0:
          variance = self._get_variance(t)
          variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device)
          pred_prev_sample += torch.sqrt(variance) * variance_noise

      return {"prev_sample": pred_prev_sample}


    def _get_variance(self, t: int):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else 1.0
        variance = self.betas[t] * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        variance = torch.clamp(variance, min=1e-20)

        if self.variance_type == "fixed_small":
            variance = self.betas[t]
        elif self.variance_type == "fixed_large":
            variance = self.betas[t] ** 2
            if t == 1:
                variance *= 1.1
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor):
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype, device=timesteps.device)
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_cumprod[timesteps])

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


def sample_images(unet, scheduler, num_samples, image_size, device):
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
