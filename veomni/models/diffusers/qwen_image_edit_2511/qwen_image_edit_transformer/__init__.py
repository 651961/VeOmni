from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_edit_transformer_config():
    from .configuration_qwen_image_edit_transformer import QwenImageEditTransformerModelConfig

    return QwenImageEditTransformerModelConfig


@MODELING_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_edit_transformer_modeling(architecture: str = None):
    from .modeling_qwen_image_edit_transformer import QwenImageEditTransformerModel

    return QwenImageEditTransformerModel
