from typing import List, Optional

from transformers import PretrainedConfig


class Krea2ConditionModelConfig(PretrainedConfig):
    """Configuration for the Krea-2 image-edit offline condition model."""

    model_type = "Krea2ConditionModel"

    def __init__(
        self,
        base_model_path: str = "/models/Krea-2-Raw",
        text_encoder_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        max_length: int = 512,
        select_layers: List[int] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        target_image_area: int = 1024 * 1024,
        condition_image_area: int = 448 * 448,
        reference_vae_image_area: int = 1024 * 1024,
        image_size_multiple: int = 32,
        num_train_timesteps: int = 1000,
        timestep_shift_min_resolution: int = 256,
        timestep_shift_max_resolution: int = 1280,
        timestep_shift_y1: float = 0.5,
        timestep_shift_y2: float = 1.15,
        timestep_shift_mu: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.text_encoder_path = text_encoder_path or f"{base_model_path}/text_encoder"
        self.tokenizer_path = tokenizer_path or f"{base_model_path}/tokenizer"
        self.vae_path = vae_path or f"{base_model_path}/vae"
        self.max_length = max_length
        self.select_layers = list(select_layers)
        self.target_image_area = target_image_area
        self.condition_image_area = condition_image_area
        self.reference_vae_image_area = reference_vae_image_area
        self.image_size_multiple = image_size_multiple
        self.num_train_timesteps = num_train_timesteps
        self.timestep_shift_min_resolution = timestep_shift_min_resolution
        self.timestep_shift_max_resolution = timestep_shift_max_resolution
        self.timestep_shift_y1 = timestep_shift_y1
        self.timestep_shift_y2 = timestep_shift_y2
        self.timestep_shift_mu = timestep_shift_mu
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("condition_model_type") == cls.model_type:
            config_dict["model_type"] = cls.model_type
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
