from typing import List, Optional, Tuple, Union
import torch 
import torch.nn as nn 
import numpy as np
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
        self.beta = beta_start
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        if self.beta_schedule == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        self.register_buffer("betas", betas)
         
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        timesteps = torch.arange(self.num_train_timesteps, dtype=torch.long)
        self.register_buffer("timesteps", timesteps)

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`."
            )
        timesteps = torch.linspace(0, self.num_train_timesteps - 1, steps=num_inference_steps, dtype=torch.long)
        self.timesteps = timesteps.to(device)

    def previous_timestep(self, timestep):
        if timestep > 0:
            prev_t = timestep - 1
        else:
            prev_t = None  
        return prev_t

    def _get_variance(self, t):
        prev_t = t - 1
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if t > 0 else 1.0
        current_beta_t = self.betas[t]
    
        variance = self.betas[t] * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        variance = torch.clamp(variance, min=1e-20)

        if self.variance_type == "fixed_small":
            variance = self.betas[t]
        elif self.variance_type == "fixed_large":
            variance = self.betas[t] ** 2
            if t == 1:
                variance = variance * 1.1
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor: 
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        t = timestep
        prev_t = self.previous_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]
        
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev)
        current_sample_coeff = torch.sqrt(1.0 - alpha_prod_t_prev)

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output

        variance = 0
        if t > 0:
            variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device)
            variance = self._get_variance(t)
        
        pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * variance_noise
        
        return pred_prev_sample
