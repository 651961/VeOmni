from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .....utils.import_utils import is_liger_kernel_available, is_package_available
from .configuration_krea2_transformer import Krea2TransformerModelConfig

if is_package_available("triton"):
    try:
        from .....ops.kernels.rotary.triton_wan import ApplyRotaryEmb
    except Exception:
        ApplyRotaryEmb = None
else:
    ApplyRotaryEmb = None

if is_liger_kernel_available():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    except ImportError:
        LigerSiLUMulFunction = None
else:
    LigerRMSNormFunction = None
    LigerSiLUMulFunction = None


@dataclass
class Krea2TransformerOutput(ModelOutput):
    loss: Optional[dict[str, torch.Tensor]] = None
    predictions: Optional[torch.Tensor] = None


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float,
    freqs_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError(f"Rotary dim must be even, got {dim}.")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos.to(dtype=freqs_dtype), freqs)
    cos = freqs.cos().float()
    sin = freqs.sin().float()
    return cos, sin


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs_cis
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)
    if ApplyRotaryEmb is not None and x.is_cuda:
        return ApplyRotaryEmb.apply(x, cos.contiguous(), sin.contiguous(), False)

    cos = cos.repeat_interleave(2, dim=-1, output_size=cos.shape[-1] * 2)[None, :, None, :]
    sin = sin.repeat_interleave(2, dim=-1, output_size=sin.shape[-1] * 2)[None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _normalize_attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.bool()
    if bool(mask.all().item()):
        return None
    return mask


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if enable_gqa and query.shape[2] % key.shape[2] != 0:
        raise ValueError(
            f"GQA requires query heads to be divisible by key/value heads, got "
            f"{query.shape[2]} query heads and {key.shape[2]} key/value heads."
        )

    attention_mask = _normalize_attention_mask(attention_mask)
    query = rearrange(query, "b s h d -> b h s d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        enable_gqa=enable_gqa,
    )
    return rearrange(out, "b h s d -> b s h d")


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: the effective multiplier is `1 + weight`, matching the Krea 2 checkpoint
    format. The activations are upcast so the normalization runs in float32; the scale weight is kept in float32 by the
    model's `_keep_in_fp32_modules`."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if LigerRMSNormFunction is not None and hidden_states.is_cuda:
            return LigerRMSNormFunction.apply(
                hidden_states,
                self.weight,
                self.eps,
                1.0,
                "gemma",
                False,
                None,
            )
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps)
        return hidden_states.to(dtype)

class Krea2SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate(hidden_states)
        up = self.up(hidden_states)
        if LigerSiLUMulFunction is not None and hidden_states.is_cuda:
            hidden_states = LigerSiLUMulFunction.apply(gate, up)
        else:
            hidden_states = F.silu(gate) * up
        return self.down(hidden_states)


class Krea2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            key = _apply_rotary_emb(key, image_rotary_emb)

        hidden_states = _attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2TextFusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class Krea2TextProjection(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2RotaryPosEmbed(nn.Module):
    _DEFAULT_CACHE_LENGTH = 8192
    _CACHE_LENGTH_GRANULARITY = 1024

    def __init__(self, theta: float, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self._freqs_cache: dict[tuple[int, str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_axis_freqs(
        self,
        axis_idx: int,
        axis_dim: int,
        ids: torch.Tensor,
        freqs_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = ids.device
        cache_key = (axis_idx, device.type, device.index, freqs_dtype)
        cached = self._freqs_cache.get(cache_key)
        cache_len = max(ids.shape[0], self._DEFAULT_CACHE_LENGTH)
        cache_len = math.ceil(cache_len / self._CACHE_LENGTH_GRANULARITY) * self._CACHE_LENGTH_GRANULARITY

        if cached is None or cached[0].shape[0] < cache_len:
            table_pos = torch.arange(cache_len, dtype=torch.float32, device=device)
            cached = _get_1d_rotary_pos_embed(axis_dim, table_pos, theta=self.theta, freqs_dtype=freqs_dtype)
            self._freqs_cache[cache_key] = cached

        axis_ids = ids[:, axis_idx].to(dtype=torch.long)
        cos, sin = cached
        return cos.index_select(0, axis_ids), sin.index_select(0, axis_ids)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_out = []
        sin_out = []
        freqs_dtype = torch.float64 if ids.device.type not in ("mps", "npu") else torch.float32
        for i, axis_dim in enumerate(self.axes_dim):
            if ids.is_floating_point():
                cos, sin = _get_1d_rotary_pos_embed(
                    axis_dim,
                    ids[:, i].float(),
                    theta=self.theta,
                    freqs_dtype=freqs_dtype,
                )
            else:
                cos, sin = self._get_axis_freqs(i, axis_dim, ids, freqs_dtype)
            cos_out.append(cos)
            sin_out.append(sin)
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2TransformerModel(PreTrainedModel):
    """VeOmni-native Krea-2 DiT.

    The module hierarchy matches the diffusers-format `transformer/`
    checkpoint, but the forward is implemented locally so training uses
    VeOmni/FSDP/gradient-checkpointing semantics and local SDPA dispatch.
    """

    config_class = Krea2TransformerModelConfig
    main_input_name = "img"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    _no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]
    _keep_in_fp32_modules = ["norm", "norm1", "norm2", "norm_q", "norm_k"]

    def __init__(self, config: Krea2TransformerModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        hidden_size = config.attention_head_dim * config.num_attention_heads
        if sum(config.axes_dims_rope) != config.attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(config.axes_dims_rope)} must equal "
                f"attention_head_dim={config.attention_head_dim}."
            )

        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.img_in = nn.Linear(config.in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(config.timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=config.num_text_layers,
            dim=config.text_hidden_dim,
            num_heads=config.text_num_attention_heads,
            num_kv_heads=config.text_num_key_value_heads,
            intermediate_size=config.text_intermediate_size,
            num_layerwise_blocks=config.num_layerwise_text_blocks,
            num_refiner_blocks=config.num_refiner_text_blocks,
            eps=config.norm_eps,
        )
        self.txt_in = Krea2TextProjection(config.text_hidden_dim, hidden_size, eps=config.norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=config.rope_theta, axes_dim=list(config.axes_dims_rope))
        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=config.in_channels, eps=config.norm_eps)

    def _krea2_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}.")

        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]
        dtype = next(self.parameters()).dtype

        hidden_states = hidden_states.to(dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)
        timestep = timestep.to(dtype=dtype)

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.bool()
            if not bool(encoder_attention_mask.all().item()):
                text_attention_mask = encoder_attention_mask[:, None, None, :]
                image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
                attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb_mod,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        return self.final_layer(hidden_states, temb)

    def forward(
        self,
        img: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        target_start: int,
        target_len: int,
        training_target: Optional[torch.Tensor] = None,
    ) -> Krea2TransformerOutput:
        if img.ndim != 3:
            raise ValueError(f"`img` must have shape (B, image_seq_len, C), got {tuple(img.shape)}.")
        if img.shape[0] != 1:
            raise ValueError(f"Krea-2 DiT SFT requires --train.micro_batch_size=1, got batch size {img.shape[0]}.")
        if context.ndim != 4:
            raise ValueError(
                f"`context` must have shape (B, text_seq_len, num_layers, text_dim), got {tuple(context.shape)}."
            )
        if pos.ndim == 3:
            if pos.shape[0] != 1:
                raise ValueError("Krea-2 SFT currently expects micro_batch_size=1 for position_ids.")
            position_ids = pos[0]
        else:
            position_ids = pos

        text_mask = mask[:, :target_start].bool() if mask is not None else None
        predictions = self._krea2_forward(
            hidden_states=img,
            encoder_hidden_states=context,
            timestep=t,
            position_ids=position_ids,
            encoder_attention_mask=text_mask,
        )
        predictions = predictions[:, :target_len]

        loss = None
        if training_target is not None:
            training_target = training_target.to(device=predictions.device, dtype=predictions.dtype)
            per_elem = F.mse_loss(predictions.float(), training_target.float(), reduction="none")
            thresh = self.config.loss_outlier_threshold
            if thresh is not None:
                outlier_mask = (predictions.float() - training_target.float()).abs() > thresh
                per_elem = per_elem.masked_fill(outlier_mask, 0.0)
            loss = {"mse_loss": per_elem.mean()}

        return Krea2TransformerOutput(loss=loss, predictions=predictions)
