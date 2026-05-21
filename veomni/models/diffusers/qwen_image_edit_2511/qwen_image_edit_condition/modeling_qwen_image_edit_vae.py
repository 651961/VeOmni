# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""3D VAE used by Qwen-Image-Edit-2511.

Causal-3D encoder / decoder. For QIE-2511 the VAE is run on still images
(temporal dimension is 1), so the feature-cache plumbing inherited from the
video-VAE reference is harmless dead code along the training-time call path
(``feat_cache=None``). It is kept verbatim so the module hierarchy and
state_dict key names match the released ``vae/`` shards exactly - the
checkpoint loads with no converter.

``encode`` returns the posterior **mean** with the per-channel
``latents_mean`` / ``latents_std`` normalisation already applied, in shape
``(B, 16, H/8, W/8)`` (4-D, 16 channels). Do not return the full 32-channel
``(mean, log-variance)`` parameters or a 5-D ``(B, 32, 1, H/8, W/8)`` tensor
- both were known preprocessing bugs in an earlier port and produce silently
broken training.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn


CACHE_T = 2


class QwenImageCausalConv3d(torch.nn.Conv3d):
    """3D causal convolution. Pads only the *past* time direction.

    Standard Conv3d with the time-axis padding flipped to be one-sided so the
    output at frame ``t`` only depends on input frames ``<= t``. The
    ``cache_x`` argument is used during streaming video inference; for image
    encode it is always ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # One-sided causal padding for the time axis; spatial axes pad symmetrically.
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = torch.nn.functional.pad(x, padding)
        return super().forward(x)


class QwenImageRMS_norm(nn.Module):
    """RMS norm with channel-or-last broadcasting and an optional bias term."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            torch.nn.functional.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
            + self.bias
        )


class QwenImageResidualBlock(nn.Module):
    """Residual block of two causal 3x3x3 convs with RMSNorm + SiLU + Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        del non_linearity  # SiLU is hardcoded.
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = torch.nn.SiLU()

        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class QwenImageAttentionBlock(nn.Module):
    """Single-head causal self-attention applied per time slice via 1x1 conv2d."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImageUpsample(nn.Upsample):
    """``nn.Upsample`` that preserves the input dtype (interpolation runs in fp32)."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    """Spatial 2x up / down sampler with optional time-axis causal 3x1x1 conv.

    Modes:
        upsample2d   - 2x nearest-exact in (H, W) then channel-halving 3x3 conv.
        upsample3d   - same plus a time-axis causal 3x1x1 conv that doubles T.
        downsample2d - zero-pad then 3x3 conv with stride 2 in (H, W).
        downsample3d - same plus a time-axis causal 3x1x1 conv with stride 2 in T.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageMidBlock(nn.Module):
    """Mid block: ResnetBlock - (AttnBlock - ResnetBlock) x num_layers."""

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageEncoder3d(nn.Module):
    """3D causal encoder. Maps RGB ``(B, 3, T, H, W)`` to ``(B, 2*z_dim, T_lat, H/8, W/8)``."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_downsample=None,
        dropout=0.0,
        non_linearity: str = "silu",
        image_channels=3,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = torch.nn.SiLU()

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(image_channels, dims[0], 3, padding=1)

        self.down_blocks = torch.nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """Decoder block: ``num_res_blocks + 1`` residual blocks then an optional upsample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)
        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    """3D causal decoder. Inverse of ``QwenImageEncoder3d``."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_upsample=None,
        dropout=0.0,
        non_linearity: str = "silu",
        image_channels=3,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.nonlinearity = torch.nn.SiLU()

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2

            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            if upsample_mode is not None:
                scale *= 2.0

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, image_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageVAE(torch.nn.Module):
    """Top-level VAE wrapper.

    State-dict layout (matches the released ``vae/`` shards exactly):

        encoder.{conv_in, down_blocks.*, mid_block.*, norm_out, conv_out}
        decoder.{conv_in, mid_block.*, up_blocks.*, norm_out, conv_out}
        quant_conv.{weight,bias}
        post_quant_conv.{weight,bias}

    ``encode(image_4d)`` returns the **already-normalised** posterior mean in
    shape ``(B, 16, H/8, W/8)``. Inside ``encode`` the input is briefly
    unsqueezed to a time-1 5-D tensor; the time dim is squeezed back out
    before returning, so the caller never sees the temporal axis.
    """

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temperal_downsample: Tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        image_channels: int = 3,
        latents_mean: Optional[List[float]] = None,
        latents_std: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if attn_scales is None:
            attn_scales = []
        if latents_mean is None:
            latents_mean = (
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            )
        if latents_std is None:
            latents_std = (
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            )

        self.z_dim = z_dim
        self.temperal_downsample = list(temperal_downsample)
        self.temperal_upsample = list(reversed(temperal_downsample))

        self.encoder = QwenImageEncoder3d(
            base_dim,
            z_dim * 2,
            list(dim_mult),
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
            image_channels=image_channels,
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(
            base_dim,
            z_dim,
            list(dim_mult),
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            image_channels=image_channels,
        )

        # Stored as plain tensors (not buffers / parameters) so they don't
        # appear in ``state_dict`` - they are constants from the config, not
        # weights. The values are baked into ``vae/config.json`` and re-loaded
        # at construction time via the keyword arguments above.
        self.mean = torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1)
        self.std = 1 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """RGB ``(B, 3, H, W)`` -> normalised latent mean ``(B, 16, H/8, W/8)``."""
        del kwargs
        x = x.unsqueeze(2)
        x = self.encoder(x)
        x = self.quant_conv(x)
        x = x[:, : self.z_dim]
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        std = self.std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * std
        x = x.squeeze(2)
        return x

    def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Normalised latent ``(B, 16, H/8, W/8)`` -> RGB ``(B, 3, H, W)``."""
        del kwargs
        x = x.unsqueeze(2)
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        std = self.std.to(dtype=x.dtype, device=x.device)
        x = x / std + mean
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        x = x.squeeze(2)
        return x
