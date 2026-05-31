from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_edit_transformer_config():
    from .configuration_qwen_image_edit_transformer import QwenImageEditTransformerModelConfig

    return QwenImageEditTransformerModelConfig


@MODELING_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_edit_transformer_modeling(architecture: str = None):
    from .checkpoint_tensor_converter import create_qwen_image_edit_fuse_qkv_converter
    from .modeling_qwen_image_edit_transformer import QwenImageEditTransformerModel

    # Merge the released split Q/K/V weights into the fused to_qkv / to_added_qkv
    # layout at load time when config.fused_qkv is True (the converter returns
    # None otherwise, so the split layout still loads strictly).
    QwenImageEditTransformerModel._create_checkpoint_tensor_converter = staticmethod(
        create_qwen_image_edit_fuse_qkv_converter
    )

    return QwenImageEditTransformerModel
