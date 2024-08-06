import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import TRUNC_VAL
from .preconditioning import (EDMPreconditioning, InitialPreconditioning,
                              NoisePreconditioning, OldNoisePreconditioning)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        vae,
        sampling_timesteps=1000,
        cfg=False,
        cfg_weight=None,
        cfg_mode="single",
        cfg_drop_prob=0.2,
        preconditioning="noise",
        sigma_data=0.5,
        sigma_min=0.01,
        sigma_max=100,
        train_noise_dist="uniform",
        P_mean=-1.2,
        P_std=1.2,
        rho=1,
        S_churn=50,
        S_min=0.05,
        S_max=10,
        S_noise=1.0,
        ode_solver="euler"
    ):
        super().__init__()

        # assert cfg_mode == "single", "Only single mode is supported for now"

        self.model = model
        self.channels = self.model.channels

        self.vae = vae

        # Classifier-free Guidance
        self.cfg = cfg
        self.cfg_weight = cfg_weight if cfg_weight is not None else 1.0
        assert cfg_mode in ["single", "instructpix2pix", "sdfusion"]
        self.cfg_mode = cfg_mode
        self.cfg_drop_prob = cfg_drop_prob if self.cfg else 0.0

        # Noise Parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Only used for the lognormal training noise distribution
        self.P_std = P_std
        self.P_mean = P_mean

        # Sampling Parameters
        self.sampling_timesteps = sampling_timesteps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        assert ode_solver in ["euler", "heun"]
        self.ode_solver = ode_solver

        # Preconditioning
        if preconditioning == "noise":
            self.preconditioning = NoisePreconditioning(model)
        elif preconditioning == "edm":
            self.preconditioning = EDMPreconditioning(model, sigma_data)
        elif preconditioning == "initial":
            self.preconditioning = InitialPreconditioning(model)
        else:
            raise ValueError(
                f"Unknown preconditioning type {preconditioning}")

        # Training Noise Distribution
        if train_noise_dist == "uniform":
            self.get_sigma = lambda shape: (
                torch.rand(shape) * (self.sigma_max - self.sigma_min) + self.sigma_min)
        elif train_noise_dist == "lognormal":
            self.get_sigma = lambda shape: (torch.randn(shape) * self.P_std + self.P_mean).exp().clamp(
                self.sigma_min, self.sigma_max)
        else:
            raise ValueError(f"Unknown train_noise_dist {train_noise_dist}")

    def set_all_sample_parameters(self, sampling_timesteps, rho, S_churn, S_min, S_max, S_noise):
        self.sampling_timesteps = sampling_timesteps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    @torch.no_grad()
    def maybe_cfg(self, x, sigma, heights, conditions, floorplans, boundaries, coarse):
        model_predictions = self.preconditioning(
            x, sigma, heights, conditions=conditions, floorplans=floorplans, boundaries=boundaries, coarse=coarse)

        if self.cfg and self.cfg_weight != 1.0:
            no_conditions = [torch.zeros_like(
                conditions[0]) for _ in conditions]
            no_heights = torch.zeros_like(
                heights) if heights is not None else None
            no_floorplans = torch.zeros_like(
                floorplans) if floorplans is not None else None
            no_boundaries = torch.zeros_like(
                boundaries) if boundaries is not None else None
            no_coarse = torch.zeros_like(
                coarse) if coarse is not None else None

            cond_free_predictions = self.preconditioning(
                x, sigma, no_heights, conditions=no_conditions, floorplans=no_floorplans, boundaries=no_boundaries, coarse=no_coarse)

            model_predictions = cond_free_predictions + \
                (model_predictions - cond_free_predictions) * self.cfg_weight

        return model_predictions

    @torch.no_grad()
    def sample(self,
               shape,
               conditions=[],
               heights=None,
               floorplans=None,
               boundaries=None,
               coarse=None,
               num_steps=None,
               show_progress=False,
               return_noise_levels=False):
        # TODO Taking shape as argument shouldn't be required when conditions are given, if none is given use some default
        assert len(conditions) == self.model.num_conditions

        device = conditions[0].device if len(
            conditions) > 0 else "cuda"  # TODO hacky
        num_steps = self.sampling_timesteps if num_steps is None else num_steps

        step_indices = torch.arange(num_steps, device=device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) *
                   (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # hack for now, replace with actual solution
        def encode_condition(cond):
            if cond is None:
                return cond

            cond, pad_0, pad_1, pad_2 = self.vae.ensure_padding(
                cond, pad_value=TRUNC_VAL, min_factor=self.model.min_factor)
            return self.vae.encode(cond).mean, pad_0, pad_1, pad_2

        if len(conditions) == 2:
            conds1, pad_0, pad_1, pad_2 = encode_condition(conditions[0])
            conds2, _, _, _ = encode_condition(conditions[1])

            conditions = [conds1, conds2]
            shape = conds1.shape
        elif len(conditions) == 4:
            # here we can have None for the conditions
            raise NotImplementedError
        else:
            pad_0, pad_1, pad_2 = 0, 0, 0

        if boundaries is not None:
            boundaries, _, _, _ = self.vae.ensure_padding(
                boundaries, pad_value=TRUNC_VAL, min_factor=self.model.min_factor)
            boundaries = self.vae.encode(boundaries).mean

        x_next = torch.randn(shape, device=device) * t_steps[0]

        if return_noise_levels:
            noise_level_samples = []

        used_t_steps = zip(t_steps[:-1], t_steps[1:])
        if show_progress:
            used_t_steps = tqdm(used_t_steps, total=len(t_steps) - 1)

        for i, (t_cur, t_next) in enumerate(used_t_steps):
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / num_steps, np.sqrt(2) -
                        1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            timestep_cond = t_hat.repeat(x_hat.shape[0])
            denoised = self.maybe_cfg(
                x_hat, timestep_cond, heights, conditions=conditions, floorplans=floorplans, boundaries=boundaries, coarse=coarse)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if self.ode_solver == "heun":
                # Apply 2nd order correction.
                if i < num_steps - 1:
                    timestep_cond = t_next.repeat(x_next.shape[0])
                    denoised = self.maybe_cfg(
                        x_next, timestep_cond, heights, conditions=conditions, floorplans=floorplans, boundaries=boundaries, coarse=coarse)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * \
                        (0.5 * d_cur + 0.5 * d_prime)

            if return_noise_levels:
                noise_level_samples.append(x_next.clone())

        res = self.vae.decode(x_next)
        depadded = res[:, :, :res.shape[-3] - pad_0,
                       :res.shape[-2] - pad_1, :res.shape[-1] - pad_2]

        if not return_noise_levels:
            return depadded
        else:
            noise_level_samples = [self.vae.decode(
                x) for x in noise_level_samples]
            noise_level_samples = [x[:, :, :x.shape[-3] - pad_0,
                                     :x.shape[-2] - pad_1, :x.shape[-1] - pad_2] for x in noise_level_samples]
            return depadded, noise_level_samples

    def p_losses(self, chunks, heights=None, conditions=[], floorplans=None, boundaries=None, semantic=None, coarse=None):
        if coarse is not None:
            assert coarse.shape[-1] == chunks.shape[-1] // 2
            assert self.cfg_drop_prob == 0.0
            assert semantic is None

        device = chunks.device

        chunks, _, _, _ = self.vae.ensure_padding(
            chunks, pad_value=TRUNC_VAL, min_factor=self.model.min_factor)
        x_start = self.vae.encode(chunks).mean

        boundaries_latent = None
        if boundaries is not None:
            boundaries, _, _, _ = self.vae.ensure_padding(
                boundaries, pad_value=TRUNC_VAL, min_factor=self.model.min_factor)
            boundaries_latent = self.vae.encode(boundaries).sample()

        cfg_iter = random.random() < self.cfg_drop_prob

        if self.cfg and cfg_iter:
            # null heights
            heights = torch.zeros_like(
                heights) if heights is not None else None

            # null floorplans
            if floorplans is not None:
                floorplans = torch.zeros_like(floorplans)

            # null boundaries
            if boundaries is not None:
                boundaries_latent = torch.zeros_like(boundaries_latent)

        def condition_preprocess(cond):
            if self.cfg and cfg_iter:
                return torch.zeros_like(x_start)

            cond, _, _, _ = self.vae.ensure_padding(
                cond, pad_value=TRUNC_VAL, min_factor=self.model.min_factor)
            cond = self.vae.encode(cond).sample()
            # cond = self.vae.encode(cond).mean
            return cond

        conditions = [condition_preprocess(cond) for cond in conditions]

        sigma_shape = (x_start.shape[0], 1, 1, 1, 1)
        sigma = self.get_sigma(sigma_shape).to(device)

        n = torch.randn_like(x_start) * sigma

        D_yn = self.preconditioning(x_start + n, sigma, heights, conditions=conditions,
                                    floorplans=floorplans, boundaries=boundaries_latent, coarse=coarse)

        if self.model.is_super_scaling:
            # use L1 loss for super scaling
            weight = 1 / torch.abs(self.preconditioning.c_out(sigma))
            loss = weight * torch.abs(D_yn - x_start)
        else:
            weight = 1 / (self.preconditioning.c_out(sigma) ** 2)
            loss = weight * ((D_yn - x_start) ** 2)

        if semantic is not None:
            semantic, _, _, _ = self.vae.ensure_padding(
                semantic, pad_value=0, min_factor=self.model.min_factor)

            sem_weights = torch.ones_like(semantic, dtype=torch.float32)
            # 0 is unknown, 1 is floor/ceiling, 2 is wall, 3 is other, 4+ is furniture
            sem_weights[semantic >= 4] = 5

            scale_factor = semantic.shape[-1] // chunks.shape[-1]
            sem_weights = nn.MaxPool3d(scale_factor, scale_factor)(sem_weights)

            # normalize sem_weights to 1 average
            sem_weights = sem_weights / torch.mean(sem_weights)

            loss = sem_weights * loss
        elif floorplans is not None:
            channel_weights = torch.arange(
                floorplans.shape[1], device=device, dtype=torch.float32)[None, :, None, None]

            has_furniture = torch.max(
                floorplans * channel_weights, dim=1)[0] >= 4

            # pool by factor between chunk_latents and floorplan
            scale_factor = floorplans.shape[-1] // chunks.shape[-1]
            has_furniture = nn.MaxPool2d(
                scale_factor, scale_factor)(has_furniture.float())

            # weigh furniture 5x, everything else 1x
            has_furniture = has_furniture * 4 + 1

            # normalize weights to 1 average
            has_furniture = has_furniture / torch.mean(has_furniture)

            loss = has_furniture[:, None, :, None, :] * loss

        loss = torch.mean(loss, dim=(1, 2, 3, 4))
        return loss, torch.flatten(sigma)

    def forward(self, sample, *args, **kwargs):
        return self.p_losses(sample, *args, **kwargs)
