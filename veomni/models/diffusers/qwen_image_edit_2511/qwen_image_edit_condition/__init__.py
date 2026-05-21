from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImageEditConditionModel")
def register_qwen_image_edit_condition_config():
    from .configuration_qwen_image_edit_condition import QwenImageEditConditionModelConfig

    return QwenImageEditConditionModelConfig


@MODELING_REGISTRY.register("QwenImageEditConditionModel")
def register_qwen_image_edit_condition_modeling(architecture: str = None):
    from .modeling_qwen_image_edit_condition import QwenImageEditConditionModel

    return QwenImageEditConditionModel
