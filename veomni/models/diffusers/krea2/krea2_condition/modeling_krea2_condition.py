from __future__ import annotations

import math
from typing import Any, List, Optional

import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor, nn
from transformers import PreTrainedModel

from .....distributed.parallel_state import get_parallel_state
from .....utils.device import get_device_type
from .configuration_krea2_condition import Krea2ConditionModelConfig
from .modeling_krea2_text_encoder import Krea2Qwen3VLConditioner


def _calculate_dimensions(target_area: int, ratio: float, multiple: int) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = max(multiple, round(width / multiple) * multiple)
    height = max(multiple, round(height / multiple) * multiple)
    return int(width), int(height)


def _resize_reference_image(image: Image.Image, target_area: int, multiple: int = 32) -> Image.Image:
    width, height = image.size
    ref_width, ref_height = _calculate_dimensions(target_area, width / height, multiple)
    return image.convert("RGB").resize((ref_width, ref_height), Image.Resampling.LANCZOS)


def _image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(height, width, 3).permute(2, 0, 1)
    data = data.to(device=device, dtype=dtype) / 127.5 - 1.0
    return data.unsqueeze(0)


def _prepare_image_tokens(img: torch.Tensor, patch: int, axis0: int):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 0] = axis0
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    return img, imgpos, imgmask


def _prepare_with_refs(target: torch.Tensor, refs: list[torch.Tensor], txtlen: int, patch: int, txtmask: torch.Tensor):
    target, target_pos, target_mask = _prepare_image_tokens(target, patch, axis0=0)
    ref_tokens, ref_pos, ref_mask = [], [], []
    for i, ref in enumerate(refs):
        tokens, pos, mask = _prepare_image_tokens(ref, patch, axis0=i + 1)
        ref_tokens.append(tokens)
        ref_pos.append(pos)
        ref_mask.append(mask)

    b = target.shape[0]
    txtpos = torch.zeros(b, txtlen, 3, device=target.device)
    img = torch.cat([target, *ref_tokens], dim=1) if ref_tokens else target
    pos = torch.cat([txtpos, target_pos, *ref_pos], dim=1) if ref_pos else torch.cat([txtpos, target_pos], dim=1)
    mask = (
        torch.cat([txtmask, target_mask, *ref_mask], dim=1)
        if ref_mask
        else torch.cat([txtmask, target_mask], dim=1)
    )
    return img, pos, mask, target.shape[1]


def _krea2_dynamic_mu(seq_len: int, x1: int, x2: int, y1: float, y2: float) -> float:
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def _krea2_shifted_timesteps(
    seq_len: int,
    steps: int,
    x1: int,
    x2: int,
    y1: float,
    y2: float,
    mu: float | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Krea-2 RAW timestep shift used by the released inference sampler.

    The official sampler builds a 1 -> 0 grid of length ``steps + 1``, shifts it
    with resolution-derived ``mu``, and uses all but the terminal 0 during the
    denoising loop. Training samples uniformly from that same shifted grid.
    """
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)[:-1]
    if mu is None:
        mu = _krea2_dynamic_mu(seq_len, x1, x2, y1, y2)
    exp_mu = math.exp(float(mu))
    ts = exp_mu / (exp_mu + (1.0 / ts - 1.0))
    return ts.to(dtype=dtype)


class _Krea2QwenAutoencoder(nn.Module):
    """Qwen-Image VAE wrapper matching the official Krea-2 inference contract."""

    compression = 8
    channels = 16

    def __init__(self, vae_path: str):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.ae = AutoencoderKLQwenImage.from_pretrained(vae_path)
        self.register_buffer("latents_mean", torch.tensor(self.ae.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latents_std", torch.tensor(self.ae.latents_std).view(1, -1, 1, 1, 1))

    def decode(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c h w -> b c 1 h w")
        x = (x * self.latents_std) + self.latents_mean
        return rearrange(self.ae.decode(x).sample, "b c 1 h w -> b c h w")

    def encode(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c h w -> b c 1 h w")
        x = self.ae.encode(x).latent_dist.mode()
        x = (x - self.latents_mean) / self.latents_std
        return rearrange(x, "b c 1 h w -> b c h w")


class Krea2ConditionModel(PreTrainedModel):
    """Frozen Krea-2 VAE and Qwen3-VL edit conditioner.

    ``get_condition`` is the offline-embedding entry point. The image resize,
    tensor conversion, VAE normalization, and prompt encoding mirror the
    released Krea-2 inference preprocessing.
    """

    config_class = Krea2ConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: Krea2ConditionModelConfig, meta_init: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.meta_init = meta_init
        self.encoder = None
        self.vae = None
        self.patch = 2
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed(self.config.seed + get_parallel_state().dp_rank)
        if not meta_init:
            self._load_components()

    @property
    def _execution_device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def _execution_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def _load_components(self) -> None:
        self.encoder = Krea2Qwen3VLConditioner(
            self.config.text_encoder_path,
            tokenizer_path=self.config.tokenizer_path,
            max_length=self.config.max_length,
            select_layers=tuple(self.config.select_layers),
        )
        self.vae = _Krea2QwenAutoencoder(self.config.vae_path)
        self.encoder = self.encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        self.vae = self.vae.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    @torch.no_grad()
    def _encode_vae_image(self, image: Image.Image, target_area: int) -> torch.Tensor:
        resized = _resize_reference_image(image, target_area, self.config.image_size_multiple)
        pixel = _image_to_tensor(resized, self._execution_device, self._execution_dtype)
        return self.vae.encode(pixel).cpu()

    @torch.no_grad()
    def get_condition(
        self,
        inputs: List[str],
        images: List[Any],
        videos: Optional[List[Any]] = None,
        outputs: Optional[List[Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        del videos, outputs, kwargs

        prompts = list(inputs)
        target_latents: list[torch.Tensor] = []
        ref_latents_per_sample: list[list[torch.Tensor]] = []
        condition_images_per_sample: list[list[Image.Image]] = []

        for sample_images in images:
            assert isinstance(sample_images, dict), (
                "Each sample's images entry must contain 'image' and 'edit_image' keys."
            )
            target = sample_images.get("image")
            if isinstance(target, list):
                assert len(target) == 1, "Krea-2 edit SFT requires exactly one target image per sample."
                target = target[0]
            assert isinstance(target, Image.Image), "Target image must be a PIL.Image."
            target_latents.append(self._encode_vae_image(target, self.config.target_image_area))

            refs = sample_images.get("edit_image") or []
            if isinstance(refs, Image.Image):
                refs = [refs]
            assert len(refs) >= 1, "Krea-2 edit SFT requires at least one reference image."

            sample_ref_latents = []
            sample_condition_images = []
            for ref in refs:
                assert isinstance(ref, Image.Image), "Reference image must be a PIL.Image."
                sample_ref_latents.append(self._encode_vae_image(ref, self.config.reference_vae_image_area))
                sample_condition_images.append(
                    _resize_reference_image(
                        ref,
                        self.config.condition_image_area,
                        self.config.image_size_multiple,
                    )
                )
            ref_latents_per_sample.append(sample_ref_latents)
            condition_images_per_sample.append(sample_condition_images)

        prompt_embeds: list[torch.Tensor] = []
        prompt_masks: list[torch.Tensor] = []
        for prompt, condition_images in zip(prompts, condition_images_per_sample):
            hidden, mask = self.encoder.forward_edit([prompt], condition_images)
            prompt_embeds.append(hidden.cpu())
            prompt_masks.append(mask.cpu())

        return {
            "latents": target_latents,
            "edit_latents": ref_latents_per_sample,
            "prompt_emb": prompt_embeds,
            "prompt_emb_mask": prompt_masks,
        }

    def process_condition(
        self,
        latents: List[torch.Tensor],
        edit_latents: List[List[torch.Tensor]],
        prompt_emb: List[torch.Tensor] | torch.Tensor,
        prompt_emb_mask: List[torch.Tensor] | torch.Tensor,
        **unused,
    ) -> dict[str, Any]:
        del unused
        assert len(latents) == 1, "Krea-2 DiT SFT requires --train.micro_batch_size=1."
        x = latents[0]
        refs = list(edit_latents[0]) if edit_latents else []
        pe = prompt_emb[0] if isinstance(prompt_emb, list) else prompt_emb
        pm = prompt_emb_mask[0] if isinstance(prompt_emb_mask, list) else prompt_emb_mask
        if pe.ndim in (2, 3):
            pe = pe.unsqueeze(0)
        if pm.ndim == 1:
            pm = pm.unsqueeze(0)

        device = x.device
        dtype = x.dtype
        timestep_id = torch.randint(
            0,
            self.config.num_train_timesteps,
            (x.shape[0],),
            device=self.generator.device,
            generator=self.generator,
        )
        target_seq_len = (x.shape[-2] // self.patch) * (x.shape[-1] // self.patch)
        align = _Krea2QwenAutoencoder.compression * self.patch
        x1 = (self.config.timestep_shift_min_resolution // align) ** 2
        x2 = (self.config.timestep_shift_max_resolution // align) ** 2
        timestep_grid = _krea2_shifted_timesteps(
            target_seq_len,
            self.config.num_train_timesteps,
            x1,
            x2,
            self.config.timestep_shift_y1,
            self.config.timestep_shift_y2,
            self.config.timestep_shift_mu,
            device=self.generator.device,
            dtype=dtype,
        )
        timestep = timestep_grid[timestep_id].to(device=device)
        noise = torch.randn(x.shape, device=self.generator.device, dtype=dtype, generator=self.generator).to(device)
        x_t = (1.0 - timestep.view(-1, 1, 1, 1)) * x + timestep.view(-1, 1, 1, 1) * noise
        training_target = noise - x
        img, pos, mask, target_len = _prepare_with_refs(x_t, refs, pe.shape[1], self.patch, pm.bool())
        training_target, _, _ = _prepare_image_tokens(training_target, self.patch, axis0=0)

        return {
            "img": img,
            "context": pe.to(device=device, dtype=dtype),
            "t": timestep,
            "pos": pos,
            "mask": mask,
            "target_start": pe.shape[1],
            "target_len": target_len,
            "training_target": training_target,
        }
