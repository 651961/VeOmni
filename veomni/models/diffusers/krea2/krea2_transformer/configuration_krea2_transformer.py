from typing import List, Optional

from transformers import PretrainedConfig


class Krea2TransformerModelConfig(PretrainedConfig):
    """Configuration for Krea-2 DiT SFT."""

    model_type = "Krea2Transformer2DModel"
    condition_model_type = "Krea2ConditionModel"

    def __init__(
        self,
        attention_head_dim: int = 128,
        axes_dims_rope: List[int] = (32, 48, 48),
        in_channels: int = 64,
        intermediate_size: int = 16384,
        norm_eps: float = 1e-5,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        num_layers: int = 28,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        num_text_layers: int = 12,
        patch_size: int = 2,
        rope_theta: float = 1000.0,
        text_hidden_dim: int = 2560,
        text_intermediate_size: int = 6912,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        timestep_embed_dim: int = 256,
        loss_outlier_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.attention_head_dim = attention_head_dim
        self.axes_dims_rope = list(axes_dims_rope)
        self.in_channels = in_channels
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_layers = num_layers
        self.num_layerwise_text_blocks = num_layerwise_text_blocks
        self.num_refiner_text_blocks = num_refiner_text_blocks
        self.num_text_layers = num_text_layers
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.text_hidden_dim = text_hidden_dim
        self.text_intermediate_size = text_intermediate_size
        self.text_num_attention_heads = text_num_attention_heads
        self.text_num_key_value_heads = text_num_key_value_heads
        self.timestep_embed_dim = timestep_embed_dim
        self.loss_outlier_threshold = loss_outlier_threshold
        super().__init__(**kwargs)
