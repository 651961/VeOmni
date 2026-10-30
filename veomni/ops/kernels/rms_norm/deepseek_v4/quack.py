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

"""Quack affine RMSNorm with library-owned autograd."""

from torch import Tensor

from ..group import group_last_dim


def wrapper(x: Tensor, weight: Tensor, *, eps: float, group_size: int | None = None) -> Tensor:
    """Normalize the last dimension, optionally as independent groups."""
    from quack.rmsnorm import rmsnorm

    shape = x.shape
    if group_size is not None:
        x = group_last_dim(x, group_size)
        weight = group_last_dim(weight, group_size)
    return rmsnorm(x, weight, eps=eps).reshape(shape)
