from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Krea2ConditionModel")
def register_krea2_condition_config():
    from .configuration_krea2_condition import Krea2ConditionModelConfig

    return Krea2ConditionModelConfig


@MODELING_REGISTRY.register("Krea2ConditionModel")
def register_krea2_condition_modeling(architecture: str = None):
    from .modeling_krea2_condition import Krea2ConditionModel

    return Krea2ConditionModel
