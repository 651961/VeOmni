from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Krea2Transformer2DModel")
def register_krea2_transformer_config():
    from .configuration_krea2_transformer import Krea2TransformerModelConfig

    return Krea2TransformerModelConfig


@MODELING_REGISTRY.register("Krea2Transformer2DModel")
def register_krea2_transformer_modeling(architecture: str = None):
    from .modeling_krea2_transformer import Krea2TransformerModel

    return Krea2TransformerModel
