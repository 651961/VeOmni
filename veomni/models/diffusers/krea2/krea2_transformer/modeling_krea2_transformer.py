from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import Krea2Transformer2DModel as DiffusersKrea2Transformer2DModel
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_krea2_transformer import Krea2TransformerModelConfig


@dataclass
class Krea2TransformerOutput(ModelOutput):
    loss: Optional[dict[str, torch.Tensor]] = None
    predictions: Optional[torch.Tensor] = None


class Krea2TransformerModel(PreTrainedModel):
    """VeOmni training wrapper for the diffusers Krea-2 transformer.

    The underlying modules are registered at this class' top level so the
    standard diffusers-format `/transformer` checkpoint loads without key
    conversion. The forward signature matches VeOmni's DiT offline-training
    cache and adds the flow-matching MSE used by SFT.
    """

    config_class = Krea2TransformerModelConfig
    main_input_name = "img"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]

    def __init__(self, config: Krea2TransformerModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        inner = DiffusersKrea2Transformer2DModel(
            in_channels=config.in_channels,
            num_layers=config.num_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            timestep_embed_dim=config.timestep_embed_dim,
            text_hidden_dim=config.text_hidden_dim,
            num_text_layers=config.num_text_layers,
            text_num_attention_heads=config.text_num_attention_heads,
            text_num_key_value_heads=config.text_num_key_value_heads,
            text_intermediate_size=config.text_intermediate_size,
            num_layerwise_text_blocks=config.num_layerwise_text_blocks,
            num_refiner_text_blocks=config.num_refiner_text_blocks,
            axes_dims_rope=tuple(config.axes_dims_rope),
            rope_theta=config.rope_theta,
            norm_eps=config.norm_eps,
        )
        for name, module in inner._modules.items():
            self.add_module(name, module)
        self.in_channels = inner.in_channels
        self.out_channels = inner.out_channels
        self.hidden_size = inner.hidden_size
        self.gradient_checkpointing = False

    def _krea2_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = DiffusersKrea2Transformer2DModel.forward.__wrapped__(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            attention_kwargs=None,
            return_dict=True,
        )
        return output.sample

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
            per_elem = F.mse_loss(predictions.float(), training_target.float(), reduction="none")
            thresh = self.config.loss_outlier_threshold
            if thresh is not None:
                outlier_mask = (predictions.float() - training_target.float()).abs() > thresh
                per_elem = per_elem.masked_fill(outlier_mask, 0.0)
            loss = {"mse_loss": per_elem.mean()}

        return Krea2TransformerOutput(loss=loss, predictions=predictions)
