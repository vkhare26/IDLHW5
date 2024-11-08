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
        """
        Args:
            num_train_timesteps (`int`): 
            
        """
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
        
    
        # TODO: calculate betas
        if self.beta_schedule == 'linear':
            # This is the DDPM implementation
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        self.register_buffer("betas", betas)
         
        # TODO: calculate alphas
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        # TODO: calculate alpha cumulative product
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        # TODO: timesteps
        timesteps = torch.arange(self.num_train_timesteps, dtype=torch.long)
        self.register_buffer("timesteps", timesteps)
        

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
            
        # TODO: set timesteps
        timesteps = torch.linspace(0, self.num_train_timesteps - 1, num=num_inference_steps, dtype=int)
        self.timesteps = torch.from_numpy(timesteps).to(device)


    def __len__(self):
        return self.num_train_timesteps


    def previous_timestep(self, timestep):
        """
        Get the previous timestep for a given timestep.
        
        Args:
            timestep (`int`): The current timestep.
        
        Return: 
            prev_t (`int`): The previous timestep.
        """
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        # TODO: caluclate previous timestep
        if timestep > 0:
         prev_t = timestep - 1
        else:
         prev_t = None  
        return prev_t

    
    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDPM. It calculates the variance $sigma_t$ for a given timestep.
        
        Args:
            t (`int`): The current timestep.
        
        Return:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """
        
        
        # TODO: calculate $beta_t$ for the current timestep using the cumulative product of alphas
        prev_t = t-1
        alpha_prod_t = self.alphas_cumprod[t]
        if t > 0:
         alpha_prod_t_prev = self.alphas_cumprod[t - 1]
        else:
         alpha_prod_t_prev = 1.0
        current_beta_t = None 
    
        # TODO: For t > 0, compute predicted variance $\beta_t$ (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        
        variance = self.betas[t] * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)


        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        # TODO: we start with two types of variance as mentioned in Section 3.2 of https://arxiv.org/pdf/2006.11239.pdf
        # 1. fixed_small: $\sigma_t = \beta_t$, this one is optimal for $x_0$ being deterministic
        # 2. fixed_large: $\sigma_t^2 = \beta$, this one is optimal for $x_0 \sim mathcal{N}(0, 1)$
        if self.variance_type == "fixed_small":
            # TODO: fixed small variance
            variance = self.betas[t]
        elif self.variance_type == "fixed_large":
            # TODO: fixed large variance
            variance = self.betas[t] ** 2
            # TODO: small hack: set the initial (log-)variance like so to get a better decoder log likelihood.
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
        """
        Add noise to the original samples. This function is used to add noise to the original samples at the beginning of each training iteration.
        
        
        Args:
            original_samples (`torch.Tensor`): 
                The original samples.
            noise (`torch.Tensor`): 
                The noise tensor.
            timesteps (`torch.IntTensor`): 
                The timesteps.
        
        Return:
            noisy_samples (`torch.Tensor`): 
                The noisy samples.
        """
        
        # make sure alphas the on the same device as samples
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        # TODO: get sqrt alphas
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod) 
        sqrt_alpha_prod = None 
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        # TODO: get sqrt one miucs alphas
        sqrt_one_minus_alpha_prod = torch.sqrt(alphas_cumprod-1) 
        sqrt_one_minus_alpha_prod = None 
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # TODO: add noise to the original samples using the formula (14) from https://arxiv.org/pdf/2006.11239.pdf
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """
        
        
        t = timestep
        prev_t = self.previous_timestep(t)
        
        # TODO: 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t-1] 
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]
        
        # TODO: 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # TODO: 3. Clip or threshold "predicted x_0" (for better sampling quality)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # TODO: 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev)
        current_sample_coeff = torch.sqrt(1.0 - alpha_prod_t_prev)

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output



        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            # TODO: use self,get_variance and variance_noise
            variance = self._get_variance(t)
        
        # TODO: add variance to prev_sample
        pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * variance_noise
        
        return pred_prev_sample