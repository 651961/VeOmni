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
"""Configuration for the Qwen-Image-Edit-2511 DiT.

Field schema matches the upstream ``transformer/config.json`` from the
released Qwen-Image-Edit-2511 checkpoint plus a few VeOmni-side training
knobs (``loss_outlier_threshold``).
"""

from typing import List, Optional

from transformers import PretrainedConfig


class QwenImageEditTransformerModelConfig(PretrainedConfig):
    """Schema for the Qwen-Image-Edit-2511 DiT.

    The default values mirror the upstream ``transformer/config.json``:

    * ``patch_size = 2`` - 2x2 spatial patchify packed inside ``img_in``.
    * ``in_channels = 64`` - patchified latent dim (16 latent channels x 2x2 patch).
    * ``out_channels = 16`` - VAE latent channels.
    * ``num_layers = 60`` - dual-stream MMDiT blocks. The last block's text
      branch is structurally dead (see Phase 8 docs).
    * ``num_attention_heads = 24``, ``attention_head_dim = 128`` -> hidden 3072.
    * ``joint_attention_dim = 3584`` - text encoder hidden size (Qwen2.5-VL).
    * ``axes_dims_rope = (16, 56, 56)`` - RoPE axis split across (time, H, W).
    * ``zero_cond_t = True`` - duplicate the batch and zero the second half's
      timestep for edit tokens (Qwen-Image-Edit-2511 specific).

    Extra VeOmni-side fields:

    * ``loss_outlier_threshold`` - elementwise outlier mask on the flow-matching
      MSE; any element with ``|prediction - target| > threshold`` is zeroed out
      before the spatial mean (bf16 stability guard). ``None`` disables.
    """

    model_type = "QwenImageTransformer2DModel"
    condition_model_type = "QwenImageEditConditionModel"

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        joint_attention_dim: int = 3584,
        axes_dims_rope: List[int] = (16, 56, 56),
        guidance_embeds: bool = False,
        zero_cond_t: bool = True,
        loss_outlier_threshold: Optional[float] = None,
        fused_qkv: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.joint_attention_dim = joint_attention_dim
        self.axes_dims_rope = list(axes_dims_rope)
        self.guidance_embeds = guidance_embeds
        self.zero_cond_t = zero_cond_t
        self.loss_outlier_threshold = loss_outlier_threshold
        # When True each stream's Q/K/V projection is a single fused GEMM
        # (``to_qkv`` for image, ``to_added_qkv`` for text) instead of three
        # separate ``nn.Linear`` calls -- fewer kernel launches and better
        # tensor-core utilisation in both forward and backward. The released
        # checkpoint's separate ``to_q/to_k/to_v`` (and ``add_{q,k,v}_proj``)
        # weights are merged into the fused weight at load time by
        # ``QwenImageEditFuseQKVConverter``. Set False to restore the original
        # split projections (e.g. for numerical-alignment debugging).
        self.fused_qkv = fused_qkv
        super().__init__(**kwargs)
