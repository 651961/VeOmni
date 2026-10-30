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

"""Quack partial RoPE for shared, duplicated-half rotary tables."""

from torch import Tensor


def wrapper(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> tuple[Tensor, Tensor]:
    """Rotate a prefix of [S,H,D] or [B,S,H,D] using fixed [S,R] tables.

    Tables contain identical halves and are cast to the activation dtype,
    matching H3. [B,H,S,D] is supported with unsqueeze_dim=1.
    """
    from quack.rotary import apply_rotary_emb

    if cos.requires_grad or sin.requires_grad:
        raise ValueError("Quack RoPE requires fixed rotary tables.")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("Quack RoPE requires shared [S,R] rotary tables.")
    if (q.ndim == 3 and unsqueeze_dim != 1) or q.ndim not in (3, 4) or unsqueeze_dim not in (1, 2):
        raise ValueError("Quack RoPE supports [S,H,D], [B,H,S,D], or [B,S,H,D].")
    half = cos.shape[-1] // 2

    def rotate(x: Tensor) -> Tensor:
        shaped = x.unsqueeze(0) if x.ndim == 3 else x.transpose(1, 2) if unsqueeze_dim == 1 else x
        out = apply_rotary_emb(
            shaped,
            cos[:, :half].to(x.dtype),
            sin[:, :half].to(x.dtype),
            interleaved=False,
            inplace=False,
        )
        return out.squeeze(0) if x.ndim == 3 else out.transpose(1, 2) if unsqueeze_dim == 1 else out

    return rotate(q), rotate(k)
