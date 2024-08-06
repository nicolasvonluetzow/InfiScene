import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from ..diffusion.utils import TRUNC_VAL

"""
Building blocks adapted from: https://github.com/lucidrains/denoising-diffusion-pytorch
"""


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=8, dropout_prob=0.5):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, padding=padding)

        groups = 1 if out_channels % groups != 0 else groups
        # self.norm = nn.GroupNorm(groups, out_channels)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

        if dropout_prob > 0:
            self.dropout = nn.Dropout3d(dropout_prob)
        else:
            self.dropout = None

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return self.dropout(x) if self.dropout is not None else x


class ResNetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, kernel_size=3, padding=1, dropout_prob=0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Conv3DBlock(
            dim, dim_out, kernel_size=kernel_size, padding=padding, dropout_prob=dropout_prob)
        self.block2 = Conv3DBlock(
            dim_out, dim_out, kernel_size=kernel_size, padding=padding, dropout_prob=dropout_prob)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv3d(dim, dim_out, 3, padding=1)
    )


def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c (w p1) (l p2) (h p3) -> b (c p1 p2 p3) w l h",
                  p1=2, p2=2, p3=2),
        nn.Conv3d(dim * 8, dim_out, 1)
    )


class ScalingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, encoder=True):
        super().__init__()

        self.block1 = ResNetBlock(
            input_channels, output_channels)

        self.block2 = ResNetBlock(
            output_channels, output_channels)

        self.scaling = Downsample(output_channels, output_channels) if encoder else Upsample(
            output_channels, output_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return self.scaling(x)


class GaussianDistribution:
    def __init__(self, parameters):
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30, 20)

        self.std = torch.exp(0.5 * self.log_var)

    def sample(self, noise=None):
        if noise is None:
            noise = torch.randn_like(self.std)
        return self.mean + self.std * noise


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=8, process_channels=64, num_scaling_blocks=2):
        super().__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.process_channels = process_channels

        self.scale_factor = 2 ** num_scaling_blocks

        self.encoder = nn.Sequential(
            ResNetBlock(input_channels, process_channels,
                        kernel_size=5, padding=2),
            *[ScalingBlock(process_channels, process_channels)
              for _ in range(num_scaling_blocks)],
            ResNetBlock(process_channels, latent_channels)
        )

        self.decoder = nn.Sequential(
            ResNetBlock(latent_channels, process_channels),
            *[ScalingBlock(process_channels, process_channels, encoder=False)
              for _ in range(num_scaling_blocks)],
            ResNetBlock(process_channels, input_channels)
        )

        self.quant_conv = nn.Conv3d(latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)

    def ensure_padding(self, x, pad_value, min_factor=1):
        # ensure that the shape of x is divisible by self.scale_factor and the shape of the latent is divisible by min_factor
        factor = min_factor * self.scale_factor
        pad_0, pad_1, pad_2 = \
            (factor - x.shape[2] % factor) % factor, \
            (factor - x.shape[3] % factor) % factor, \
            (factor - x.shape[4] % factor) % factor

        x = nn.functional.pad(
            x, (0, pad_2, 0, pad_1, 0, pad_0), mode='constant', value=pad_value)

        return x, pad_0, pad_1, pad_2

    def __is_valid_shape__(self, x):
        return x.shape[-3] % self.scale_factor == 0 and \
            x.shape[-2] % self.scale_factor == 0 and \
            x.shape[-1] % self.scale_factor == 0

    def encode(self, x):
        assert self.__is_valid_shape__(x)
        x = self.encoder(x)
        return GaussianDistribution(self.quant_conv(x))

    def decode(self, latent):
        assert len(latent.shape) == 5
        x = self.post_quant_conv(latent)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x, pad_0, pad_1, pad_2 = self.ensure_padding(x, TRUNC_VAL)

        gaussian = self.encode(x)

        z = gaussian.sample()

        recon = self.decode(z)

        recon = recon[:, :, :recon.shape[2] - pad_0,
                      :recon.shape[3] - pad_1, :recon.shape[4] - pad_2]

        return recon, gaussian.mean, gaussian.log_var


class NoGaussianDistribution:
    def __init__(self, latent):
        self.mean = latent

    def sample(self, noise=None):
        return self.mean


class NoAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=1, process_channels=0, num_scaling_blocks=0):
        super().__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.process_channels = process_channels

        self.scale_factor = 1

    def ensure_padding(self, x, pad_value, min_factor=1):
        # ensure that the shape of x is divisible by self.scale_factor and the shape of the latent is divisible by min_factor
        factor = min_factor * self.scale_factor
        pad_0, pad_1, pad_2 = \
            (factor - x.shape[2] % factor) % factor, \
            (factor - x.shape[3] % factor) % factor, \
            (factor - x.shape[4] % factor) % factor

        x = nn.functional.pad(
            x, (0, pad_2, 0, pad_1, 0, pad_0), mode='constant', value=pad_value)

        return x, pad_0, pad_1, pad_2

    def __is_valid_shape__(self, x):
        return x.shape[-3] % self.scale_factor == 0 and \
            x.shape[-2] % self.scale_factor == 0 and \
            x.shape[-1] % self.scale_factor == 0

    def encode(self, x):
        x = torch.clamp(x, 0, TRUNC_VAL)
        x = (x / TRUNC_VAL) * 2 - 1
        return NoGaussianDistribution(x)

    def decode(self, latent):
        assert len(latent.shape) == 5
        latent = (latent + 1) / 2 * TRUNC_VAL
        latent = torch.clamp(latent, 0, TRUNC_VAL)
        return latent

    def forward(self, x):
        return x, x, torch.zeros_like(x)
