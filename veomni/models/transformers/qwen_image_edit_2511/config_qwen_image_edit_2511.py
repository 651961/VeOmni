# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

import json
import os

from transformers import PretrainedConfig


try:
    import diffusers

    diffusers_version = diffusers.__version__

except ModuleNotFoundError:
    raise ImportError("diffusers is not installed")


class QwenImageEdit2511Config(PretrainedConfig):
    model_type = "qwen_image_edit_2511"

    def __init__(
        self,
        attention_head_dim=128,
        axes_dims_rope=[16, 56, 56],
        guidance_embeds=False,
        in_channels=64,
        joint_attention_dim=3584,
        num_attention_heads=24,
        num_layers=60,
        out_channels=16,
        patch_size=2,
        use_additional_t_cond=False,
        use_layer3d_rope=False,
        zero_cond_t=True,
        pooled_projection_dim=768,
        **kwargs,
    ):
        self.attention_head_dim = attention_head_dim
        self.axes_dims_rope = axes_dims_rope
        self.guidance_embeds = guidance_embeds
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_additional_t_cond = use_additional_t_cond
        self.use_layer3d_rope = use_layer3d_rope
        self.zero_cond_t = zero_cond_t
        self.pooled_projection_dim = pooled_projection_dim

        super().__init__(**kwargs)

    # @classmethod
    # def from_pretrained(cls, path):
    #     config = AutoConfig.from_pretrained(path)
    #     return cls(**config.to_dict())

    def save_pretrained(self, path):
        config = {
            "_class_name": "QwenImageTransformer2DModel",
            "_diffusers_version": diffusers_version,
            "attention_head_dim": self.attention_head_dim,
            "axes_dims_rope": self.axes_dims_rope,
            "guidance_embeds": self.guidance_embeds,
            "in_channels": self.in_channels,
            "joint_attention_dim": self.joint_attention_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_layers": self.num_layers,
            "out_channels": self.out_channels,
            "patch_size": self.patch_size,
            "use_additional_t_cond": self.use_additional_t_cond,
            "use_layer3d_rope": self.use_layer3d_rope,
            "zero_cond_t": self.zero_cond_t,
            "model_type": "qwen_image_edit_2511",
        }

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
