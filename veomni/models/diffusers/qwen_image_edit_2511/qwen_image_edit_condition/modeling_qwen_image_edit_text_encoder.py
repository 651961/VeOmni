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
"""Text encoder used by Qwen-Image-Edit-2511.

Constructs ``Qwen2_5_VLModel`` from a literal ``Qwen2_5_VLConfig`` dict and
loads the released ``text_encoder/`` shards manually after rewriting key
prefixes (``visual.* -> model.visual.*``, ``model.* ->
model.language_model.*``).

Rationale - this manual construction path is bit-exact against the
reference SFT prompt-embedding cache; loading the same checkpoint via
``Qwen2_5_VLForConditionalGeneration.from_pretrained`` (or
``Qwen2_5_VLModel.from_pretrained``) gives ~100 absolute diff on the final
hidden state despite identical state_dict, buffers, module list and
config (the divergence has not been root-caused; the manual path is the
known-good workaround). Keep the literal config inlined - reading
``config.json`` from disk participates in the same divergence.

The encoder is image-aware: the prompt-edit path passes ``pixel_values``
and ``image_grid_thw`` so that the edit image's vision tokens land in
the right slots before the language tower runs.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Union

import torch
from safetensors.torch import load_file


# Literal config dict used to construct ``Qwen2_5_VLModel`` for the
# Qwen-Image-Edit-2511 text encoder. Matches the released checkpoint's
# ``config.json`` field-for-field; inlined here because reading the file
# from disk participates in the numerical divergence described in the
# module docstring.


def _convert_state_dict(state_dict: dict) -> dict:
    """Rewrite checkpoint key prefixes to match ``Qwen2_5_VLModel``.

    Released checkpoint keys are laid out as
    ``visual.*`` (vision tower) and ``model.*`` (language tower);
    ``Qwen2_5_VLModel`` expects ``model.visual.*`` and
    ``model.language_model.*``. ``lm_head.*`` is left untouched - it
    lands on our placeholder ``lm_head`` linear (never called by the
    SFT path which only consumes ``hidden_states``).
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.")
        out[k] = v
    return out


def _load_shards(path: str) -> dict:
    """Load all safetensors shards under ``path`` into a single dict."""
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "*.safetensors")))
    sd: dict = {}
    for shard in shards:
        sd.update(load_file(os.path.join(path, shard)))
    return sd


class QwenImageEditTextEncoder(torch.nn.Module):
    """Image-aware text encoder for the Qwen-Image-Edit-2511 SFT recipe.

    Constructs the encoder from the literal config dict above and loads
    weights via :func:`safetensors.torch.load_file` + key rewriting. The
    extra ``lm_head`` placeholder gives the checkpoint's ``lm_head.weight``
    a home; the SFT path never invokes it (we only consume the per-layer
    ``hidden_states`` tuple via ``forward``'s ``[-1]`` index downstream).
    """

    def __init__(self) -> None:
        super().__init__()
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel
        config = Qwen2_5_VLConfig(**{
            "architectures": [
                "Qwen2_5_VLForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_token_id": 151655,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 128000,
            "max_window_layers": 28,
            "model_type": "qwen2_5_vl",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "rope_type": "default",
                "type": "default"
            },
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "text_config": {
                "architectures": [
                    "Qwen2_5_VLForConditionalGeneration"
                ],
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "hidden_act": "silu",
                "hidden_size": 3584,
                "image_token_id": None,
                "initializer_range": 0.02,
                "intermediate_size": 18944,
                "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
                ],
                "max_position_embeddings": 128000,
                "max_window_layers": 28,
                "model_type": "qwen2_5_vl_text",
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "rope_type": "default",
                "type": "default"
                },
                "rope_theta": 1000000.0,
                "sliding_window": None,
                "torch_dtype": "float32",
                "use_cache": True,
                "use_sliding_window": False,
                "video_token_id": None,
                "vision_end_token_id": 151653,
                "vision_start_token_id": 151652,
                "vision_token_id": 151654,
                "vocab_size": 152064
            },
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "transformers_version": "4.54.0",
            "use_cache": True,
            "use_sliding_window": False,
            "video_token_id": 151656,
            "vision_config": {
                "depth": 32,
                "fullatt_block_indexes": [
                    7,
                    15,
                    23,
                    31
                ],
                "hidden_act": "silu",
                "hidden_size": 1280,
                "in_channels": 3,
                "in_chans": 3,
                "initializer_range": 0.02,
                "intermediate_size": 3420,
                "model_type": "qwen2_5_vl",
                "num_heads": 16,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
                "tokens_per_second": 2,
                "torch_dtype": "float32",
                "window_size": 112
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "vocab_size": 152064
        })
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "QwenImageEditTextEncoder":
        """Load from a released ``text_encoder/`` directory of safetensors.

        ``kwargs`` is accepted for call-site parity but ignored - all
        construction details (config, attn impl) flow through the
        literal-config path; ``attn_implementation`` is whatever
        transformers chooses for the freshly constructed
        ``Qwen2_5_VLModel`` (sdpa by default on supported torch builds).
        """
        del kwargs
        instance = cls()
        sd = _load_shards(pretrained_model_name_or_path)
        converted = _convert_state_dict(sd)
        # strict=False because the checkpoint's ``lm_head.weight`` lands on
        # our placeholder ``lm_head`` (which IS in ``instance``'s
        # state_dict); but any other unexpected/missing keys would still
        # surface here, so log them for visibility.
        missing, unexpected = instance.load_state_dict(converted, strict=False)
        if missing:
            # Filter out keys we know don't exist in our placeholder layout
            # (none expected; surface honestly if anything appears).
            raise RuntimeError(f"text_encoder load: missing keys {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            raise RuntimeError(
                f"text_encoder load: unexpected keys {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )
        if torch_dtype is not None:
            instance = instance.to(dtype=torch_dtype)
        return instance

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        """Return the underlying model's hidden-state tuple.

        ``output_attentions`` is forced False; ``output_hidden_states``
        forced True. ``logits_to_keep`` / ``labels`` are accepted for API
        parity and ignored (no logits returned).
        """
        del logits_to_keep, labels
        output_attentions = False
        output_hidden_states = True

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs.hidden_states
