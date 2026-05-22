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
"""Qwen-Image-Edit-2511 DiT.

Dual-stream MMDiT with rectified-flow conditioning. Implemented from scratch
on plain ``torch.nn`` modules. The module hierarchy is preserved exactly so
the released ``transformer/`` shards load via ``load_state_dict(strict=True)``
with no converter.

The forward pass consolidates the steps the reference SFT recipe runs across
``model_fn_qwen_image``:

    1. Patchify the noisy main latent ``(B, 16, H/8, W/8)`` -> token packing
       ``(B, (H/16)*(W/16), 64)`` (2x2 patch packs 16 channels x 2 x 2 = 64).
    2. If edit latents are present, patchify each and concatenate along the
       token dim. The combined ``img_shapes`` list drives the rotary
       embedding shape and the post-attention token slicing.
    3. ``img_in`` lifts 64 -> 3072. ``txt_in(txt_norm(prompt_emb))`` lifts
       3584 -> 3072 on the text side.
    4. ``zero_cond_t`` doubles the batch on ``timestep`` and zeros the second
       copy; ``modulate_index`` is built so target tokens see the real
       timestep and edit-condition tokens see ``t=0`` (Qwen-Image-Edit-2511
       specific).
    5. 60 dual-stream blocks: each does joint cross-attention over
       ``[text, image]`` with modulation derived from ``time_text_embed``.
    6. After the last block, the text stream is discarded (its last-block
       branch is structurally dead; the released checkpoint records this).
    7. ``norm_out`` applies the final timestep-modulated affine, ``proj_out``
       projects 3072 -> 64, the slice keeps only the main-image tokens, and
       the inverse rearrange returns ``(B, 16, H/8, W/8)``.

Floating-point order in the rotary-embedding cache and in the modulation
arithmetic is preserved so the 1.7 numerical-alignment gate passes (max abs
diff < 1e-2 vs the reference under bf16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .....ops.kernels.rotary.triton_wan import ApplyRotaryEmb, apply_rotary_interleaved
from .....utils import logging
from .....utils.import_utils import is_liger_kernel_available
from .configuration_qwen_image_edit_transformer import QwenImageEditTransformerModelConfig


logger = logging.get_logger(__name__)


try:
    import flash_attn_interface
    from flash_attn_interface import _flash_attn_forward as _fa3_op_fwd

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    _fa3_op_fwd = None


# --------------------------------------------------------------------------- #
# Dynamo-traceable wrappers for the rotary triton kernel and FA3 forward.
#
# flash_attn_3::_flash_attn_forward is already registered upstream as a
# torch.library.custom_op (with register_fake + register_autograd), so
# calling it directly bypasses the FlashAttnFunc autograd.Function wrapper
# (a dynamo black box) and lets dynamo trace the attention call.
#
# apply_rotary_interleaved (the triton kernel call) is wrapped below as
# veomni::rotary_interleaved_fwd with a fake meta + autograd that mirrors
# ApplyRotaryEmb's conjugate-pass.  dynamo treats this custom_op as
# opaque, matching the original autograd.Function semantics.
# --------------------------------------------------------------------------- #


def _fa3_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FA3 forward, dynamo-traceable. Inputs (B, S, H, D); returns (B, S, H, D)."""
    softmax_scale = q.shape[-1] ** (-0.5)
    out, _lse, _accum, _lse_accum = _fa3_op_fwd(
        q, k, v,
        None, None,        # k_new, v_new
        None,              # qv
        None,              # out_
        None, None, None,  # cu_seqlens_q/k/k_new
        None, None,        # seqused_q/k
        None, None,        # max_seqlen_q/k
        None, None, None,  # page_table, kv_batch_idx, leftpad_k
        None, None, None,  # rotary_cos/sin, seqlens_rotary
        None, None, None,  # q_descale, k_descale, v_descale
        softmax_scale,
    )
    return out


@torch.library.custom_op("veomni::rotary_interleaved_fwd", mutates_args=())
def _rotary_interleaved_fwd_op(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return apply_rotary_interleaved(x, cos, sin, conjugate=False)


@_rotary_interleaved_fwd_op.register_fake
def _rotary_interleaved_fwd_fake(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# Separate op for the reverse-direction (conjugate=True) triton call. Needed
# because AOT autograd Python-calls the register_autograd backward during
# joint-trace with FakeTensors, and a bare triton kernel call would try to
# access .data_ptr() on the fakes.  Wrapping it as an opaque custom_op makes
# AOT use the fake meta instead.
@torch.library.custom_op("veomni::rotary_interleaved_bwd", mutates_args=())
def _rotary_interleaved_bwd_op(
    grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return apply_rotary_interleaved(grad_out, cos, sin, conjugate=True)


@_rotary_interleaved_bwd_op.register_fake
def _rotary_interleaved_bwd_fake(
    grad_out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(grad_out)


def _rotary_interleaved_setup_ctx(ctx, inputs, output):
    _x, cos, sin = inputs
    ctx.save_for_backward(cos, sin)


def _rotary_interleaved_bwd(ctx, grad_out):
    cos, sin = ctx.saved_tensors
    grad_x = _rotary_interleaved_bwd_op(grad_out, cos, sin)
    return grad_x, None, None


_rotary_interleaved_fwd_op.register_autograd(
    _rotary_interleaved_bwd, setup_context=_rotary_interleaved_setup_ctx
)


if is_liger_kernel_available():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
else:
    LigerRMSNormFunction = None


# --------------------------------------------------------------------------- #
# Timestep / norm primitives                                                  #
# --------------------------------------------------------------------------- #


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding used by the SFT recipe.

    ``scale = 1000`` and a normalised timestep in ``[0, 1]`` together give the
    same effective embedding magnitude as feeding the raw ``[0, 1000)`` value
    through a ``scale = 1`` projection - the recipe routes through ``[0, 1]``
    so the rest of the forward stays in a friendly range.
    """
    assert timesteps.ndim == 1, "Timesteps must be a 1D tensor."
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class _TimestepEmbedder(nn.Module):
    """Two-layer MLP that turns a sinusoidal timestep embedding into the
    conditioning vector (state-dict prefix: ``timestep_embedder.linear_*``)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim_in, dim_out)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class TimeTextEmbed(nn.Module):
    """Top-level timestep embedding head, exposed as ``time_text_embed``.

    State-dict prefix: ``time_text_embed.timestep_embedder.linear_{1,2}.*``.
    """

    def __init__(self, dim_in: int = 256, dim_out: int = 3072):
        super().__init__()
        self.dim_in = dim_in
        self.timestep_embedder = _TimestepEmbedder(dim_in, dim_out)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        time_emb = _get_timestep_embedding(timestep, self.dim_in).to(dtype)
        return self.timestep_embedder(time_emb)


class _RMSNorm(nn.Module):
    """RMSNorm with an optional elementwise-affine ``weight`` parameter.

    Variance is computed in fp32 even when the input is bf16/fp16 (the
    reference recipe relies on this for stability in the joint-attention
    head normalisation).
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Fused-kernel path: same fp32-variance / weight-in-input-dtype contract
        # as the eager branch below ("llama" casting), but a single Triton kernel
        # instead of ~7 small ops. in_place=False so views surviving activation
        # checkpoint recompute are not aliased.
        if LigerRMSNormFunction is not None and self.weight is not None and hidden_states.is_cuda:
            return LigerRMSNormFunction.apply(
                hidden_states,
                self.weight,
                self.eps,
                0.0,
                "llama",
                False,
                None,
            )
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


class _AdaLayerNormSingle(nn.Module):
    """Final-layer ada-LN: ``norm(x) * (1 + scale) + shift``.

    State-dict prefix: ``<name>.linear.{weight,bias}``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(F.silu(emb))
        scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
        return self.norm(x) * (1 + scale) + shift


# --------------------------------------------------------------------------- #
# Rotary embedding                                                            #
# --------------------------------------------------------------------------- #


def _rope_params(index: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Pre-compute complex rotary frequencies for a single axis."""
    assert dim % 2 == 0
    freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis) -> torch.Tensor:
    """Multiply ``x`` by the per-axis rotary frequencies.

    ``freqs_cis`` may be either:
      * a complex tensor of shape ``(S, D/2)`` -- eager complex-multiply path,
      * a ``(cos, sin)`` real-tensor tuple of shape ``(S, D/2)`` each, in
        which case the fused interleaved-RoPE Triton kernel is used via the
        ``veomni::rotary_interleaved_fwd`` custom_op (dynamo-traceable).
        The kernel's ``(B, S, H, D)`` layout is reached by a stride-only
        ``transpose(1, 2)`` view over the ``(B, H, S, D)`` input; the same
        transpose is undone on the way out.
    The two paths are mathematically identical up to bf16 rounding.
    """
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        x_bshd = x.transpose(1, 2)
        out = _rotary_interleaved_fwd_op(x_bshd, cos, sin)
        return out.transpose(1, 2)
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    """3-axis rotary embedding shared by main and edit image tokens.

    The cache is keyed by ``(layer_idx, height, width)`` so repeated forward
    calls with the same image shape reuse the same frequency table.
    """

    def __init__(self, theta: float = 10000.0, axes_dim: Optional[List[int]] = None, scale_rope: bool = True):
        super().__init__()
        if axes_dim is None:
            axes_dim = [16, 56, 56]
        self.theta = theta
        self.axes_dim = list(axes_dim)
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([_rope_params(pos_index, self.axes_dim[i], self.theta) for i in range(3)], dim=1)
        self.neg_freqs = torch.cat([_rope_params(neg_index, self.axes_dim[i], self.theta) for i in range(3)], dim=1)
        self.rope_cache: dict = {}
        self.scale_rope = scale_rope

    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
        _, height, width = video_fhw
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        cur_max_len = self.pos_freqs.shape[0]
        if required_len <= cur_max_len:
            return
        new_max_len = math.ceil(required_len / 512) * 512
        pos_index = torch.arange(new_max_len)
        neg_index = torch.arange(new_max_len).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([_rope_params(pos_index, self.axes_dim[i], self.theta) for i in range(3)], dim=1)
        self.neg_freqs = torch.cat([_rope_params(neg_index, self.axes_dim[i], self.theta) for i in range(3)], dim=1)

    def forward(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        return vid_freqs, txt_freqs


# --------------------------------------------------------------------------- #
# Block primitives                                                            #
# --------------------------------------------------------------------------- #


class _ApproximateGELU(nn.Module):
    """``proj(x) * sigmoid(1.702 * proj(x))`` (an approximate GELU variant).

    State-dict prefix: ``<name>.proj.{weight,bias}``.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class QwenFeedForward(nn.Module):
    """Per-token MLP. State-dict prefix: ``<name>.net.{0.proj,2}.*``.

    ``net`` is a ``ModuleList`` of [GELU-proj(linear), Dropout, Linear]; the
    dropout module has no parameters, so ``.0`` and ``.2`` are the two linear
    layers in the state-dict.
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * 4)
        out_dim = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList(
            [
                _ApproximateGELU(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, out_dim),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def _flash_or_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Joint-attention call: FA3 fast path with SDPA fallback.

    Inputs are ``(B, H, S, D)`` (heads-first, matching the stream-concat
    convention in ``QwenDoubleStreamAttention.forward``). Output is
    ``(B, S, H*D)`` so the downstream per-stream slice + linear projection
    are unchanged from the previous SDPA call.

    The fast path goes through ``_fa3_attn``, which dispatches directly to
    the ``flash_attn_3::_flash_attn_forward`` custom_op (registered upstream
    with fake meta + autograd).  When ``flash_attn_interface`` is unavailable
    or ``attention_mask`` is not ``None`` we drop into SDPA, which accepts
    the bool/float mask and matches the kernel output up to bf16 rounding.
    Both branches are dynamo-traceable, so the caller can be wrapped with
    ``torch.compile`` directly.
    """
    if _FLASH_ATTN_AVAILABLE and attention_mask is None:
        query = rearrange(query, "b h s d -> b s h d")
        key = rearrange(key, "b h s d -> b s h d")
        value = rearrange(value, "b h s d -> b s h d")
        out = _fa3_attn(query, key, value)
        return rearrange(out, "b s h d -> b s (h d)")
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
    return rearrange(out, "b h s d -> b s (h d)")


class QwenDoubleStreamAttention(nn.Module):
    """Dual-stream joint attention.

    Image and text streams each have their own Q/K/V projections (``to_*`` for
    image, ``add_*_proj`` for text) and head-norms (``norm_q/k`` for image,
    ``norm_added_q/k`` for text). The two streams are concatenated along the
    token dim before the SDPA call, then split back to separate output
    projections (``to_out.0`` and ``to_add_out``).

    State-dict prefix:
        attn.to_{q,k,v}.{weight,bias}
        attn.norm_{q,k}.weight
        attn.add_{q,k,v}_proj.{weight,bias}
        attn.norm_added_{q,k}.weight
        attn.to_out.0.{weight,bias}     # ``to_out`` is a Sequential([Linear])
        attn.to_add_out.{weight,bias}
    """

    def __init__(self, dim_a: int, dim_b: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Image stream.
        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = _RMSNorm(head_dim, eps=1e-6)
        self.norm_k = _RMSNorm(head_dim, eps=1e-6)

        # Text stream.
        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = _RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = _RMSNorm(head_dim, eps=1e-6)

        self.to_out = nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, "b s (h d) -> b h s d", h=self.num_heads)
        img_k = rearrange(img_k, "b s (h d) -> b h s d", h=self.num_heads)
        img_v = rearrange(img_v, "b s (h d) -> b h s d", h=self.num_heads)
        txt_q = rearrange(txt_q, "b s (h d) -> b h s d", h=self.num_heads)
        txt_k = rearrange(txt_k, "b s (h d) -> b h s d", h=self.num_heads)
        txt_v = rearrange(txt_v, "b s (h d) -> b h s d", h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = _apply_rotary_emb(img_q, img_freqs)
            img_k = _apply_rotary_emb(img_k, img_freqs)
            txt_q = _apply_rotary_emb(txt_q, txt_freqs)
            txt_k = _apply_rotary_emb(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        joint_out = _flash_or_sdpa_attention(joint_q, joint_k, joint_v, attention_mask=attention_mask)

        txt_out = joint_out[:, :seq_txt, :]
        img_out = joint_out[:, seq_txt:, :]
        img_out = self.to_out(img_out)
        txt_out = self.to_add_out(txt_out)
        return img_out, txt_out


class QwenImageTransformerBlock(nn.Module):
    """One dual-stream block with image and text mod / attn / mlp paths.

    The ``img_mod`` / ``txt_mod`` modules each produce six scale+shift+gate
    triplets per token: three for the attention input and three for the MLP
    input. ``zero_cond_t`` doubles the batch for the image side; the
    ``modulate_index`` argument selects which half of the modulation each
    token sees (target tokens see the real timestep, edit-condition tokens
    see ``t=0``).
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim, dim_b=dim, num_heads=num_attention_heads, head_dim=attention_head_dim
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)

    def _modulate(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)
            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modulate_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        if modulate_index is not None:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated, text=txt_modulated, image_rotary_emb=image_rotary_emb, attention_mask=attention_mask
        )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out
        return text, image


# --------------------------------------------------------------------------- #
# Top-level model                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class QwenImageEditTransformerOutput(ModelOutput):
    """Model output container.

    Attributes:
        loss: Dict with key ``"mse_loss"`` holding the (already weighted)
            flow-matching MSE if ``training_target`` was provided to forward.
            Empty dict otherwise.
        predictions: Noise prediction in latent space, shape
            ``(B, 16, H/8, W/8)``.
    """

    loss: Optional[dict] = None
    predictions: Optional[torch.Tensor] = None


class QwenImageEditTransformerModel(PreTrainedModel):
    """Dual-stream MMDiT for Qwen-Image-Edit-2511.

    State-dict layout matches the released ``transformer/`` shards exactly:

        img_in.{weight,bias}                          # 64 -> 3072
        txt_in.{weight,bias}                          # 3584 -> 3072
        txt_norm.weight                               # RMSNorm(3584)
        time_text_embed.timestep_embedder.linear_{1,2}.{weight,bias}
        transformer_blocks.0..59.<the per-block keys listed in
            QwenImageTransformerBlock / QwenDoubleStreamAttention>
        norm_out.linear.{weight,bias}                 # 3072 -> 6144 (single ada-LN)
        proj_out.{weight,bias}                        # 3072 -> 64
    """

    config_class = QwenImageEditTransformerModelConfig
    supports_gradient_checkpointing = True
    # ``PreTrainedModel.__init__`` checks this flag when a config requests
    # ``attn_implementation="flash_attention_2"`` and raises otherwise. The
    # actual attention dispatch in this module is ``_flash_or_sdpa_attention``,
    # which selects FA3 (``flash_attn_interface.flash_attn_func``) when
    # available and falls back to SDPA — the flag exists only to keep the
    # v5 init-time gate from rejecting the config.
    _supports_flash_attn_2 = True
    # ``build_parallelize_model`` (veomni/distributed/torch_parallelize.py)
    # reads this attribute to find per-block FSDP2 shard targets. Without it
    # FSDP2 falls back to wrapping the whole model as a single unit, which on
    # 20B materialises ~40GB of unsharded params on every rank during the
    # block loop and OOMs immediately. The class name here must match the
    # block class created inside ``transformer_blocks``.
    _no_split_modules = ["QwenImageTransformerBlock"]

    def __init__(self, config: QwenImageEditTransformerModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        dim = config.num_attention_heads * config.attention_head_dim  # 24 * 128 = 3072
        if config.in_channels != 64 or config.patch_size != 2:
            # The patch/in_channels invariant guards against silent layout drift:
            # ``img_in`` takes the (latent_channels * patch_size * patch_size)
            # packed token. For Qwen-Image-Edit-2511 that's 16 * 2 * 2 = 64.
            logger.warning_once(
                f"Unusual config: in_channels={config.in_channels}, patch_size={config.patch_size}. "
                f"The packed token dim must equal in_channels."
            )

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=config.axes_dims_rope, scale_rope=True)
        self.time_text_embed = TimeTextEmbed(dim_in=256, dim_out=dim)
        self.txt_norm = _RMSNorm(config.joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(config.in_channels, dim)
        self.txt_in = nn.Linear(config.joint_attention_dim, dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = _AdaLayerNormSingle(dim)
        self.proj_out = nn.Linear(dim, config.out_channels * config.patch_size * config.patch_size)
        # Toggled to True by ``PreTrainedModel.gradient_checkpointing_enable``
        # (transformers v5). The block loop in ``forward`` reads this flag
        # and routes through ``self._gradient_checkpointing_func`` when set;
        # HF also requires at least one module in the tree to declare this
        # attribute or the enable call raises.
        self.gradient_checkpointing = False

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        edit_latents: Optional[List[torch.Tensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        training_target: Optional[torch.Tensor] = None,
        training_weight: Optional[torch.Tensor] = None,
    ) -> QwenImageEditTransformerOutput:
        """Run the DiT forward.

        Args:
            latents: Noisy main latent ``x_t``, shape ``(B, 16, H/8, W/8)``,
                where ``B`` is the micro-batch and ``H/W`` are the resized
                target image dimensions.
            timestep: Per-sample timestep tensor of shape ``(B,)``. Integer
                ``[0, 1000)`` range; divided by 1000 inside.
            prompt_emb: Prompt embedding of shape ``(B, L, 3584)``.
            prompt_emb_mask: Token mask of shape ``(B, L)`` int64; non-zero
                entries are valid prompt tokens, zero entries are padding.
            edit_latents: Optional list of edit-image latents, each shape
                ``(B, 16, He/8, We/8)``. Concatenated along the token dim
                after patchifying; the final output discards their tokens.
            height, width: Spatial dimensions of the *target image* before
                VAE downsample (i.e. ``H = 8 * latents.shape[2]``). When not
                given they are inferred from ``latents``.
            training_target: Optional ``(B, 16, H/8, W/8)`` target velocity
                field. If given, the per-sample flow-matching MSE is
                computed and returned under ``loss["mse_loss"]``.
            training_weight: Optional scalar tensor multiplied into the loss
                (the bsmntw weight from the scheduler at ``timestep``).

        Returns:
            QwenImageEditTransformerOutput with ``predictions`` (always) and
            ``loss`` (only when ``training_target`` is provided).
        """
        if height is None:
            height = latents.shape[2] * 8
        if width is None:
            width = latents.shape[3] * 8

        # 1) Patchify the noisy main latent into the token sequence.
        img_shapes = [(1, latents.shape[2] // 2, latents.shape[3] // 2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        # Normalise timestep to ``[0, 1)`` before the sinusoidal embedding;
        # the embedding undoes this via ``scale=1000``.
        timestep = timestep / 1000

        image = rearrange(
            latents,
            "(B N) C (H P) (W Q) -> B (N H W) (C P Q)",
            H=height // 16,
            W=width // 16,
            P=2,
            Q=2,
            N=1,
        )
        image_seq_len = image.shape[1]

        # 2) Append edit-image tokens (one per reference image) along the
        # token dim. ``img_shapes`` keeps a (1, H/16, W/16) entry per layer
        # so rotary embedding picks the right axis split.
        if edit_latents is not None:
            edit_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            img_shapes += [(e.shape[0], e.shape[2] // 2, e.shape[3] // 2) for e in edit_list]
            edit_tokens = [
                rearrange(
                    e,
                    "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=e.shape[2] // 2,
                    W=e.shape[3] // 2,
                    P=2,
                    Q=2,
                )
                for e in edit_list
            ]
            image = torch.cat([image] + edit_tokens, dim=1)

        # 3) Lift image and text to the joint hidden dim.
        image = self.img_in(image)

        # 4) zero_cond_t: duplicate the timestep batch and build the
        # per-token modulation index. Target tokens (main image) see the
        # real timestep at index 0; edit-condition tokens see t=0 at
        # index 1.
        if self.config.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            target_token_count = math.prod(img_shapes[0])
            edit_token_count = sum(math.prod(s) for s in img_shapes[1:])
            modulate_index = torch.tensor(
                [[0] * target_token_count + [1] * edit_token_count],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        conditioning = self.time_text_embed(timestep, image.dtype)

        # 5) Text and image rotary embedding.
        text = self.txt_in(self.txt_norm(prompt_emb))
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        # Decompose the complex rotary tables into real ``(cos, sin)`` pairs
        # once here so each block can dispatch the fused interleaved-RoPE
        # Triton kernel directly. ``.real`` / ``.imag`` are stride-2 views of
        # the interleaved complex storage; ``.contiguous()`` materialises
        # each as a (S, D/2) real tensor, avoiding 60 per-call copies inside
        # the kernel. CPU keeps the complex-multiply eager path.
        if latents.is_cuda:
            _vid_freqs, _txt_freqs = image_rotary_emb
            image_rotary_emb = (
                (_vid_freqs.real.contiguous(), _vid_freqs.imag.contiguous()),
                (_txt_freqs.real.contiguous(), _txt_freqs.imag.contiguous()),
            )

        # 6) 60 dual-stream blocks. When gradient checkpointing is on we
        # hand each block to ``self._gradient_checkpointing_func`` (bound by
        # ``PreTrainedModel.gradient_checkpointing_enable``); positional args
        # match ``QwenImageTransformerBlock.forward(image, text, temb,
        # image_rotary_emb, attention_mask, modulate_index)``.
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                text, image = self._gradient_checkpointing_func(
                    block,
                    image,
                    text,
                    conditioning,
                    image_rotary_emb,
                    None,
                    modulate_index,
                )
            else:
                text, image = block(
                    image=image,
                    text=text,
                    temb=conditioning,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=None,
                    modulate_index=modulate_index,
                )

        # 7) Final norm + projection. The text stream is discarded (the
        # released checkpoint's last block has structurally-dead text-side
        # parameters - this is expected).
        if self.config.zero_cond_t:
            conditioning = conditioning.chunk(2, dim=0)[0]
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)

        # 8) Drop edit-condition tokens and rearrange back to latent shape.
        image = image[:, :image_seq_len]
        predictions = rearrange(
            image,
            "B (N H W) (C P Q) -> (B N) C (H P) (W Q)",
            H=height // 16,
            W=width // 16,
            P=2,
            Q=2,
            B=1,
        )

        # 9) Loss (only when a target is provided). Threshold-masked outlier
        # handling is gated by ``loss_outlier_threshold`` on the config.
        loss = None
        if training_target is not None:
            per_elem = F.mse_loss(predictions.float(), training_target.float(), reduction="none")
            thresh = self.config.loss_outlier_threshold
            if thresh is not None:
                outlier_mask = (predictions.float() - training_target.float()).abs() > thresh
                per_elem = per_elem.masked_fill(outlier_mask, 0.0)
            mse = per_elem.mean()
            if training_weight is not None:
                mse = mse * training_weight
            loss = {"mse_loss": mse}

        return QwenImageEditTransformerOutput(loss=loss, predictions=predictions)
