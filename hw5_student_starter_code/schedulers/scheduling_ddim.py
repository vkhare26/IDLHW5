from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from utils import randn_tensor
from .scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_inference_steps is not None, "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)

    def _get_variance(self, t):
        """
        Calculate variance $sigma_t$ for a given timestep (DDIM-specific).

        Args:
            t (`int`): The current timestep.

        Returns:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """
        # Calculate cumulative products of alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        if t > 0:
            alpha_prod_t_prev = self.alphas_cumprod[t - 1]
        else:
            alpha_prod_t_prev = 1.0

        # Variance formula for DDIM
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE (DDIM method).

        Args:
            model_output (`torch.Tensor`): The direct output from the learned diffusion model.
            timestep (`int`): The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`): A current instance of a sample created by the diffusion process.
            eta (`float`): The weight of the noise to add to the variance.
            generator (`torch.Generator`, optional): A random number generator.

        Returns:
            prev_sample (`torch.Tensor`): The predicted sample at the previous timestep.
        """
        t = timestep
        prev_t = t - 1 if t > 0 else None

        # Compute alphas and betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t is not None else 1.0
        beta_prod_t = 1.0 - alpha_prod_t

        # Compute predicted original sample (x_0) from predicted noise
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # Clip or threshold predicted original sample
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # Compute variance and standard deviation for noise
        variance = self._get_variance(t)
        std_dev_t = torch.sqrt(variance)

        # Compute the directional update
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output

        # Compute x_t without random noise
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # Add noise with eta
        if eta > 0 and prev_t is not None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            prev_sample = prev_sample + eta * std_dev_t * variance_noise

        return prev_sample
