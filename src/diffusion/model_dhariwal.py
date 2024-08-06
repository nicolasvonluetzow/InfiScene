import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (avg_pool_nd, checkpoint, conv_nd, linear, normalization,
                 timestep_embedding, zero_module)

"""
Adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, conditions=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, MultiCrossAttentionBlock):
                x = layer(x, conditions)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels,
                                self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # if self.dims == 3:
        #     x = F.interpolate(
        #         x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
        #     )
        # else:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2  # if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class MultiModule(nn.Module):
    def __init__(self, module, num_modules, *args, **kwargs):
        super().__init__()

        self.module_list = nn.ModuleList([
            module(*args, **kwargs) for _ in range(num_modules)
        ])

    def forward(self, list_of_inputs):
        return [
            module_entry(input_entry) for module_entry, input_entry in zip(self.module_list, list_of_inputs)
        ]
        # for module_entry, input_entry in zip(self.module_list, list_of_inputs):
        #     input_entry = module_entry(input_entry)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels,
                        self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class CrossAttentionBlock(nn.Module):
    """
    A cross-attention block that allows conditioning using other chunks.

    Adapted from the AttentionBlock implementation. Requires x and condition to be of the same dimensions.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.q = conv_nd(1, channels, channels, 1)

        self.norm = normalization(channels)
        self.kv = conv_nd(1, channels, channels * 2, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond):
        return checkpoint(self._forward, (x, cond), self.parameters(), True)

    def _forward(self, x, cond):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        cond = cond.reshape(b, c, -1)

        q = self.q(self.norm(cond))
        kv = self.kv(self.norm(x))
        qkv = th.cat([q, kv], dim=1)

        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class MultiCrossAttentionBlock(nn.Module):
    def __init__(self, dims, num_modules, channels, *args, **kwargs):
        super().__init__()

        # self.cross_attention_modules = nn.ModuleList([
        #     CrossAttentionBlock(*args, **kwargs) for _ in range(num_modules)
        # ])

        self.collapse_conditions = conv_nd(
            dims, num_modules * channels, channels, 1, padding=0
        )

        self.cross_attention_module = CrossAttentionBlock(
            channels, *args, **kwargs)

    def forward(self, x, conditions):
        # for cross_attention_block, condition in zip(self.cross_attention_modules, conditions):
        #     x = cross_attention_block(x, condition)
        # return x

        conditions = th.cat(conditions, dim=1)
        conditions = self.collapse_conditions(conditions)
        return self.cross_attention_module(x, conditions)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3,
                              length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        cross_attention_resolutions=None,
        num_conditions=0,
        condition_channels=None,
        add_conditions_to_input=False,
        add_flipped_conditions=[],
        use_floorplan_conditions=False,
        floorplan_downsamples=1,
        floorplan_classes=10,
        use_boundary_conditions=False,
        use_height_conditioning=False,
        is_super_scaling=False
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.channels = in_channels
        self.min_factor = 2 ** (len(channel_mult) - 1)

        self.input_size = input_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.cross_attention_resolutions = cross_attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_conditions = num_conditions
        self.add_conditions_to_input = add_conditions_to_input
        self.add_flipped_conditions = add_flipped_conditions
        self.use_floorplan_conditions = use_floorplan_conditions
        self.floorplan_downsample_factor = floorplan_downsamples
        self.floorplan_classes = floorplan_classes
        self.use_boundary_conditions = use_boundary_conditions
        self.use_height_conditioning = use_height_conditioning
        self.is_super_scaling = is_super_scaling

        if is_super_scaling:
            self.coarse_upsample = Upsample(self.in_channels, True, dims)
            self.in_channels = in_channels = in_channels * 2

        if add_conditions_to_input:
            assert num_conditions > 0
            assert condition_channels is not None
            self.in_channels = in_channels = in_channels + \
                num_conditions * condition_channels

        if add_flipped_conditions is not None and add_flipped_conditions != []:
            assert condition_channels is not None
            assert len(add_flipped_conditions) == num_conditions

            self.in_channels = in_channels = in_channels + \
                len(add_flipped_conditions) * condition_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if use_height_conditioning:
            self.height_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.use_floorplan_conditions:
            # self.floorplan_preprocess = ResBlock(
            #     floorplan_classes, time_embed_dim, dropout=0, dims=2, use_scale_shift_norm=use_scale_shift_norm)
            self.floorplan_preprocess = TimestepEmbedSequential(
                conv_nd(2, floorplan_classes, model_channels, 1, padding=0),
                # ResBlock(model_channels, time_embed_dim, dropout=0,
                #          dims=2, use_scale_shift_norm=use_scale_shift_norm)
            )

            # self.floorplan_downsample = nn.Sequential(
            #     *[conv_nd(2, model_channels, model_channels, 2, stride=2, padding=0)
            #       #   if i != floorplan_downsamples - 1
            #       #   else conv_nd(2, model_channels, condition_channels, 2, stride=2, padding=0)
            #       for i in range(floorplan_downsamples)]
            # )
            self.floorplan_downsample = TimestepEmbedSequential(
                *[ResBlock(model_channels, time_embed_dim, dropout=0, use_scale_shift_norm=use_scale_shift_norm, down=True)
                  for i in range(floorplan_downsamples)]
            )

            # TODO make this depend on num res blocks
            self.floorplan_3d_preprocess = TimestepEmbedSequential(
                ResBlock(model_channels, time_embed_dim, dropout=dropout,
                         dims=3, use_scale_shift_norm=use_scale_shift_norm, kernel_size=7, padding=3),
                ResBlock(model_channels, time_embed_dim, dropout=dropout,
                         dims=3, use_scale_shift_norm=use_scale_shift_norm, kernel_size=3, padding=1),
                ResBlock(model_channels, time_embed_dim, dropout=dropout,
                         dims=3, use_scale_shift_norm=use_scale_shift_norm, kernel_size=3, padding=1),
                ResBlock(model_channels, time_embed_dim, dropout=dropout,
                         dims=3, use_scale_shift_norm=use_scale_shift_norm, kernel_size=3, padding=1),
                ResBlock(model_channels, time_embed_dim, dropout=dropout,
                         dims=3, use_scale_shift_norm=use_scale_shift_norm, kernel_size=3, padding=1),
                conv_nd(3, model_channels, condition_channels, 1, padding=0)
            )

            num_conditions += 1
            self.in_channels = in_channels = in_channels + condition_channels

        if use_boundary_conditions:
            num_conditions += 1
            self.in_channels = in_channels = in_channels + condition_channels

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(
                conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in cross_attention_resolutions:
                    layers.append(
                        MultiCrossAttentionBlock(
                            dims,
                            num_conditions,
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.downsample_indices = []

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in cross_attention_resolutions:
                    layers.append(
                        MultiCrossAttentionBlock(
                            dims,
                            num_conditions,
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        if cross_attention_resolutions is not None and cross_attention_resolutions != []:
            self.input_condition_blocks = MultiModule(
                conv_nd, num_conditions, dims, condition_channels, input_ch, 3, padding=1)

            self.max_downsamples_needed = int(math.log(
                max(cross_attention_resolutions), 2))
            self.condition_downsamples = nn.ModuleList(
                [MultiModule(
                    Downsample,
                    num_conditions,
                    int(channel_mult[level] * model_channels),
                    use_conv=conv_resample,
                    dims=dims,
                    out_channels=int(channel_mult[level + 1] * model_channels))
                 for level in range(self.max_downsamples_needed)]
            )

    def forward(self, x, timesteps, heights=None, conditions=[], floorplans=None, boundaries=None, coarse=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        assert x.shape[0] == timesteps.shape[0]
        assert heights is None or x.shape[0] == heights.shape[0]
        assert len(conditions) == self.num_conditions

        hs = []
        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels))

        if self.use_height_conditioning:
            assert heights is not None
            emb += self.height_embed(timestep_embedding(
                heights, self.model_channels))

        if self.is_super_scaling:
            assert x.shape[-1] == coarse.shape[-1] * \
                2 and x.shape[1] == coarse.shape[1]
            coarse = self.coarse_upsample(coarse)
            x = th.cat([x, coarse], dim=1)

        if self.add_conditions_to_input:
            x = th.cat([x] + conditions, dim=1)

        if self.add_flipped_conditions is not None and self.add_flipped_conditions != []:
            # we add 2 dimensions to dim as we have batch and channel dimensions
            flipped_conditions = [th.flip(condition, (dim + 2,)) for condition, dim in zip(
                conditions, self.add_flipped_conditions)]

            x = th.cat([x] + flipped_conditions, dim=1)

        if self.use_floorplan_conditions:
            assert floorplans is not None and floorplans.dim() == 4
            floorplans = self.floorplan_preprocess(
                floorplans.type(self.dtype), emb)
            floorplans = self.floorplan_downsample(floorplans, emb)
            # floorplans = self.floorplan_downsample(floorplans.type(self.dtype))

            # repeat floorplan to match height of 3d x
            floorplans = floorplans.unsqueeze(
                3).repeat(1, 1, 1, x.shape[-2], 1)
            floorplans = self.floorplan_3d_preprocess(
                floorplans.type(self.dtype), emb)
            # add floorplan to x
            x = th.cat([x, floorplans.clone()], dim=1)
            conditions = conditions + [floorplans]

        if self.use_boundary_conditions:
            assert boundaries is not None
            x = th.cat([x, boundaries], dim=1)
            conditions = conditions + [boundaries]

        scaled_conditions = [None for _ in range(len(self.channel_mult))]
        if self.cross_attention_resolutions is not None and self.cross_attention_resolutions != []:
            scaled_conditions[0] = self.input_condition_blocks(conditions)

            for level, downsample in enumerate(self.condition_downsamples, start=1):
                scaled_conditions[level] = downsample(
                    scaled_conditions[level - 1])

            do_condition_downsample = True
        else:
            do_condition_downsample = False

        h = x.type(self.dtype)

        condition_index = 0
        for module in self.input_blocks:
            pre_size = h.shape[self.dims - 1]
            h = module(h, emb, scaled_conditions[condition_index])
            hs.append(h)

            if do_condition_downsample and h.shape[self.dims - 1] != pre_size:
                condition_index += 1

        h = self.middle_block(h, emb)

        condition_index = self.max_downsamples_needed if do_condition_downsample else 0
        for module in self.output_blocks:
            pre_size = h.shape[self.dims - 1]

            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, scaled_conditions[condition_index])

            if do_condition_downsample and h.shape[self.dims - 1] != pre_size:
                condition_index -= 1

        h = h.type(x.dtype)

        # for block in self.post:
        #     h = block(h, emb)

        return self.out(h)
