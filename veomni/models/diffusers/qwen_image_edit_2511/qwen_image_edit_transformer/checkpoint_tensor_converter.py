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

"""Runtime checkpoint tensor converter for Qwen-Image-Edit-2511.

Merges the released diffusers attention layout (separate Q/K/V projections)
into the fused single-GEMM layout used when ``config.fused_qkv`` is True, at
weight-load time. This runs inside ``load_model_weights`` on full CPU tensors
*before* FSDP2 sharding, so no DTensor handling is needed here.

    Released diffusers layout (per block, image + text stream):
        transformer_blocks.{i}.attn.to_q.{weight,bias}        [D_a, D_a] / [D_a]
        transformer_blocks.{i}.attn.to_k.{weight,bias}
        transformer_blocks.{i}.attn.to_v.{weight,bias}
        transformer_blocks.{i}.attn.add_q_proj.{weight,bias}  [D_b, D_b] / [D_b]
        transformer_blocks.{i}.attn.add_k_proj.{weight,bias}
        transformer_blocks.{i}.attn.add_v_proj.{weight,bias}

    Fused target layout:
        transformer_blocks.{i}.attn.to_qkv.{weight,bias}        [3*D_a, D_a] / [3*D_a]
        transformer_blocks.{i}.attn.to_added_qkv.{weight,bias}  [3*D_b, D_b] / [3*D_b]

The merged tensor is ``cat([q, k, v], dim=0)`` -- the same [q, k, v] order the
forward pass slices back out via ``chunk(3, dim=-1)``.  The inverse split is
done at export time in ``scripts/merge_dcp_to_diffusers.py`` so the exported
diffusers checkpoint keeps the released (inference-compatible) layout.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from .....utils import logging
from ....checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Image stream: ...attn.to_{q,k,v}.{weight,bias}
_IMG_PATTERN = re.compile(r"^(.*\.attn)\.to_(q|k|v)\.(weight|bias)$")
# Text stream: ...attn.add_{q,k,v}_proj.{weight,bias}
_TXT_PATTERN = re.compile(r"^(.*\.attn)\.add_(q|k|v)_proj\.(weight|bias)$")

# Fused output name per stream.
_FUSED_NAME = {"img": "to_qkv", "txt": "to_added_qkv"}


class QwenImageEditFuseQKVConverter:
    """Buffers per-stream Q/K/V tensors and emits the fused weight/bias.

    Q/K/V for a given (block, stream, param) stream from the safetensor files
    in arbitrary order; the converter holds them until all three arrive, then
    concatenates along dim 0 and emits the fused tensor. ``weight`` and
    ``bias`` are tracked independently so each fuses its own three-tensor set.
    """

    def __init__(self):
        # {(prefix, stream, param): {"q": t, "k": t, "v": t}}
        self._buf: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_IMG_PATTERN.match(name) or _TXT_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _IMG_PATTERN.match(name)
        stream = "img"
        if match is None:
            match = _TXT_PATTERN.match(name)
            stream = "txt"
        if match is None:
            return None

        prefix, qkv, param = match.groups()  # e.g. ("...attn", "q", "weight")
        key = (prefix, stream, param)
        slot = self._buf.setdefault(key, {})
        slot[qkv] = tensor

        if len(slot) < 3:
            return None  # still accumulating

        del self._buf[key]
        merged = torch.cat([slot["q"], slot["k"], slot["v"]], dim=0)
        fused_name = f"{prefix}.{_FUSED_NAME[stream]}.{param}"
        return ConvertedCheckpointTensor(name=fused_name, tensor=merged)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if self._buf:
            unflushed = {f"{p}|{s}|{t}": sorted(v.keys()) for (p, s, t), v in self._buf.items()}
            raise RuntimeError(
                "QwenImageEdit fuse-qkv converter: incomplete q/k/v buffers "
                f"(each needs q, k, v): {unflushed}"
            )
        return []


def create_qwen_image_edit_fuse_qkv_converter(model):
    """Factory registered on the model class via ``_create_checkpoint_tensor_converter``.

    Returns ``None`` when ``config.fused_qkv`` is False so the released split
    layout loads directly via strict ``load_state_dict``.
    """
    if not getattr(model.config, "fused_qkv", True):
        return None
    return QwenImageEditFuseQKVConverter()
