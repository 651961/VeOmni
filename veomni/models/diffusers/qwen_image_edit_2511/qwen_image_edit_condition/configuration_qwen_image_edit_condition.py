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
"""Configuration for the Qwen-Image-Edit-2511 condition model.

The condition model groups the non-DiT components (tokenizer, image
processor, text encoder, VAE) plus the preprocessing knobs that control how
target / edit images are resized before VAE encoding and how prompts are
joined with the edit image for text encoding.
"""

from typing import Optional

from transformers import PretrainedConfig


class QwenImageEditConditionModelConfig(PretrainedConfig):
    """Schema for the Qwen-Image-Edit-2511 condition model.

    Args:
        base_model_path: Root of the released model layout. Subfolders are
            looked up relative to this path: ``text_encoder/``, ``vae/``,
            ``tokenizer/``, ``processor/``, ``scheduler/``.
        tokenizer_subfolder, processor_subfolder, text_encoder_subfolder,
            vae_subfolder, scheduler_subfolder: Subfolder names.
        num_train_timesteps: Length of the flow-matching schedule used during
            training (1000 in the SFT recipe).
        target_max_pixels: Target image is resized so ``height * width`` is at
            most this many pixels. ``1024 * 1024 = 1048576`` by default.
        target_size_alignment: Target image side lengths are rounded to a
            multiple of this many pixels (``32`` by default).
        vae_image_area, vae_image_alignment: Same idea, applied to images
            before the VAE encode (defaults: ``1024 * 1024`` area, ``32``
            alignment).
        condition_image_area, condition_image_alignment: Same idea, applied to
            the edit image before the text encoder ingests it as a vision
            input (defaults: ``1024 * 1024`` area, ``32`` alignment). NOTE:
            the older VeOmni copy used ``384 * 384`` here, which was a
            preprocessing bug that the SFT recipe explicitly fixes.
        zero_cond_t: Whether the DiT duplicates the batch and zeros the
            second half's timestep for edit tokens. ``True`` for
            Qwen-Image-Edit-2511.
        seed: Optional RNG seed for noise / timestep sampling.
    """

    model_type = "QwenImageEditConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        tokenizer_subfolder: str = "tokenizer",
        processor_subfolder: str = "processor",
        text_encoder_subfolder: str = "text_encoder",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        num_train_timesteps: int = 1000,
        target_max_pixels: int = 1024 * 1024,
        target_size_alignment: int = 16,
        vae_image_area: int = 1024 * 1024,
        vae_image_alignment: int = 32,
        condition_image_area: int = 384 * 384,
        condition_image_alignment: int = 32,
        zero_cond_t: bool = True,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.tokenizer_subfolder = tokenizer_subfolder
        self.processor_subfolder = processor_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.num_train_timesteps = num_train_timesteps
        self.target_max_pixels = target_max_pixels
        self.target_size_alignment = target_size_alignment
        self.vae_image_area = vae_image_area
        self.vae_image_alignment = vae_image_alignment
        self.condition_image_area = condition_image_area
        self.condition_image_alignment = condition_image_alignment
        self.zero_cond_t = zero_cond_t
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
