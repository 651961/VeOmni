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
"""Qwen-Image-Edit-2511 condition model.

Groups the non-DiT components - tokenizer, image processor, text encoder,
VAE - plus the preprocessing entry points used by the offline-embedding and
online-training paths. The flow-matching scheduler is created here too so
that the training step can sample noise / timesteps and produce the
``training_target`` and per-sample ``training_weight`` directly.

This module deliberately does **not** import diffusers: the VAE is the
hand-rolled :class:`QwenImageVAE` in :mod:`.modeling_qwen_image_edit_vae`,
the text encoder is the literal-config :class:`QwenImageEditTextEncoder`
wrapper around :class:`transformers.Qwen2_5_VLModel` (see that module's
docstring for why we don't go through ``from_pretrained``), and the
scheduler is the template-dispatched
:class:`veomni.schedulers.flow_match.FlowMatchScheduler` with
``template="Qwen-Image"``.

VAE encode contract.

The VAE here returns the already-normalised mean in shape
``(B, 16, H/8, W/8)`` (4-D, 16 channels). This is **different** from the
older VeOmni copy which stored the full ``(B, 32, 1, H/8, W/8)`` posterior
(mean + log-variance, 5-D) and ran ``DiagonalGaussianDistribution.mode()`` +
``latents_mean``/``latents_std`` normalisation later - that route had two
known correctness bugs (5-D shape + delayed normalisation) that the SFT
recipe explicitly avoids.

Text encode contract (single edit image).

For QIE-2511 each sample has *exactly one* edit image, so the prompt-edit
encoder uses the single-image template (the image tokens are inline in the
user message) and the ``drop_idx = 64`` prefix length. The multi-image
``"Picture {n}:"`` template is deliberately not used here - it has a
different ``drop_idx`` and produces a different hidden-state length than the
1-image template the SFT recipe was trained against.
"""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

from .....distributed.parallel_state import get_parallel_state
from .....schedulers.flow_match import FlowMatchScheduler
from .....utils import logging
from .....utils.device import get_device_type
from .configuration_qwen_image_edit_condition import QwenImageEditConditionModelConfig
from .modeling_qwen_image_edit_text_encoder import QwenImageEditTextEncoder
from .modeling_qwen_image_edit_vae import QwenImageVAE


logger = logging.get_logger(__name__)


# Prompt template used by the SFT recipe.
#
# Even with a single edit image the recipe routes through the multi-image
# variant: when ``edit_image`` is given as a ``list`` from the data pipeline,
# the reference dispatch picks the multi-image template (``Picture {n}:``
# image-prompt prefix per image, then the user prompt). The vision tokens
# do **not** sit inline in the user message - they live in the
# ``Picture {n}:`` prefix.
#
# ``drop_idx`` is the fixed-length template prefix to drop from the encoded
# hidden states; for this template + a single edit image its value is 64.
#
# (An earlier mis-read of the recipe used the single-image template - vision
# tokens inline in the user message. That template is exercised by the
# reference's ``encode_prompt_edit`` (single PIL.Image) path, but the SFT
# pipeline never enters it because the dataset always loads ``edit_image``
# as a list.)
_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
_PROMPT_DROP_IDX = 64


def _calculate_dimensions(target_area: int, ratio: float, alignment: int) -> tuple[int, int]:
    """Resize ``(width, height)`` so width*height ~= target_area and both are
    multiples of ``alignment``. Keeps aspect ratio (``width / height ==
    ratio`` up to alignment quantisation). Used by the auto-resize step for
    edit images (round-half-to-even alignment, default PIL.resize).
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = max(alignment, round(width / alignment) * alignment)
    height = max(alignment, round(height / alignment) * alignment)
    return width, height


def _image_crop_and_resize(
    image: "Image.Image",
    max_pixels: int,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
) -> "Image.Image":
    """First-pass resize used by the SFT data pipeline.

    Equivalent to the upstream training-stack's ``ImageCropAndResize``:

        1. If ``W * H > max_pixels``: scale uniformly so the area is
           ``<= max_pixels`` (``scale = sqrt(W*H / max_pixels)``), with
           ``int()`` truncation on each side.
        2. Floor each side to the nearest multiple of its division factor.
        3. Resize via ``torchvision.transforms.functional.resize`` with
           BILINEAR interpolation (``scale = max(target_W/W, target_H/H)`` -
           same scalar applied to both axes, the larger of the two), then
           ``center_crop`` to the final ``(target_H, target_W)``.

    Returns a fresh PIL image; the original is not mutated. ``BILINEAR + center_crop``
    here is load-bearing for cache parity vs the reference SFT recipe -
    swapping to ``PIL.resize(BICUBIC)`` gives visually-identical but
    bitwise-different pixels.
    """
    import torchvision.transforms.functional as F_tv
    from torchvision.transforms.functional import InterpolationMode

    width, height = image.size
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height = int(height / scale)
        width = int(width / scale)
    target_h = (height // height_division_factor) * height_division_factor
    target_w = (width // width_division_factor) * width_division_factor

    w0, h0 = image.size
    scale = max(target_w / w0, target_h / h0)
    image = F_tv.resize(
        image,
        (round(h0 * scale), round(w0 * scale)),
        interpolation=InterpolationMode.BILINEAR,
    )
    image = F_tv.center_crop(image, (target_h, target_w))
    return image


def _auto_resize_to_area(
    image: "Image.Image",
    target_area: int,
    alignment: int = 32,
) -> "Image.Image":
    """Second-pass resize used by the SFT recipe for edit images.

    Recomputes ``(W, H)`` so ``W * H ~= target_area`` and each side is a
    multiple of ``alignment`` (round-half-to-even). Then ``PIL.resize`` with
    its default resampling. Used:

      * for edit images before VAE encode (``target_area = 1024 * 1024``,
        ``alignment = 32``);
      * for edit images before the prompt-edit text encoder
        (``target_area = 384 * 384``, ``alignment = 32``).
    """
    width, height = image.size
    new_w, new_h = _calculate_dimensions(target_area, width / height, alignment)
    return image.resize((new_w, new_h))


class QwenImageEditConditionModel(PreTrainedModel):
    """Bundles the Qwen-Image-Edit-2511 tokenizer / processor / text-encoder /
    VAE / scheduler. Exposes the per-sample preprocessing entry point
    :meth:`get_condition` so both the offline-embedding saver and the
    online-training DiT wrapper can drive it the same way.
    """

    config_class = QwenImageEditConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: QwenImageEditConditionModelConfig, meta_init: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.meta_init = meta_init

        self.tokenizer = None
        self.processor = None
        self.text_encoder: Optional[QwenImageEditTextEncoder] = None
        self.vae: Optional[QwenImageVAE] = None
        # Scheduler is template-fixed; the training driver calls
        # ``set_timesteps(num_train_timesteps, training=True)`` once before
        # the first sampling step.
        self.scheduler = FlowMatchScheduler(template="Qwen-Image")
        # Per-rank deterministic RNG for noise / timestep sampling. Adding
        # ``dp_rank`` produces distinct sampling streams across data-parallel
        # replicas while keeping each rank reproducible from ``seed``.
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed(self.config.seed + get_parallel_state().dp_rank)
        self._timesteps_ready = False

        self._load_components()

    @property
    def _execution_device(self) -> torch.device:
        return self.vae.encoder.conv_in.weight.device

    # ------------------------------------------------------------------ #
    # Component loaders                                                  #
    # ------------------------------------------------------------------ #

    def _build_vae(self) -> QwenImageVAE:
        """Build a :class:`QwenImageVAE` configured from the released
        ``vae/config.json`` and load its weights.
        """
        import json

        base = self.config.base_model_path
        vae_dir = f"{base}/{self.config.vae_subfolder}"
        with open(f"{vae_dir}/config.json") as f:
            vae_cfg = json.load(f)
        vae = QwenImageVAE(
            base_dim=vae_cfg["base_dim"],
            z_dim=vae_cfg["z_dim"],
            dim_mult=tuple(vae_cfg["dim_mult"]),
            num_res_blocks=vae_cfg["num_res_blocks"],
            attn_scales=vae_cfg.get("attn_scales", []),
            temperal_downsample=tuple(vae_cfg["temperal_downsample"]),
            dropout=vae_cfg.get("dropout", 0.0),
            latents_mean=vae_cfg["latents_mean"],
            latents_std=vae_cfg["latents_std"],
        )
        if not self.meta_init:
            from safetensors.torch import load_file

            sd = load_file(f"{vae_dir}/diffusion_pytorch_model.safetensors")
            missing, unexpected = vae.load_state_dict(sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"VAE state_dict mismatch under strict=True: missing={missing}, unexpected={unexpected}"
                )
        vae.eval()
        return vae

    def _load_components(self) -> None:
        """Lazy-load tokenizer, processor, text encoder, VAE.

        ``meta_init=True`` is the offline-training mode: the condition model
        is never invoked at step time (we read pickled embeddings from the
        cache), so the text encoder is not loaded. The VAE is built but its
        weights are not loaded (saves ~3 GB host RAM).
        """
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Qwen-Image-Edit-2511 condition components from {base}.")

        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        # The ``subfolder=`` kwarg on ``AutoProcessor.from_pretrained`` silently
        # falls back to the tokenizer-only ``Qwen2Tokenizer`` when the
        # subfolder contains ``preprocessor_config.json`` - the processor
        # class auto-discovery only walks the top-level path. Pass the full
        # path so we get the real multimodal ``Qwen2VLProcessor`` (which
        # owns the image processor that expands ``<|image_pad|>``).
        self.processor = AutoProcessor.from_pretrained(os.path.join(base, self.config.processor_subfolder))
        self.vae = self._build_vae()
        # bf16 for the real-weights path. The VAE's ``mean``/``std`` are
        # plain (non-parameter, non-buffer) tensors that get cast at encode
        # time to match the input dtype, so ``.to(bf16)`` on the module is
        # sufficient to drive a bf16 encode end-to-end. Skip the cast in
        # meta-init mode - the random-init weights are fp32 placeholders
        # that the offline-training driver never invokes.
        if not self.meta_init:
            self.vae = self.vae.to(torch.bfloat16)

        if self.meta_init:
            self.text_encoder = None
        else:
            # bf16 to match the SFT recipe and downstream DiT dtype.
            # Uses the literal-config + manual state_dict load path inside
            # ``QwenImageEditTextEncoder`` - that path is bit-exact against
            # the reference SFT prompt-embedding cache, whereas
            # ``Qwen2_5_VLForConditionalGeneration.from_pretrained`` on the
            # same checkpoint diverges by ~100 on the final hidden state
            # (see the text-encoder module docstring for details).
            self.text_encoder = QwenImageEditTextEncoder.from_pretrained(
                os.path.join(base, self.config.text_encoder_subfolder),
                torch_dtype=torch.bfloat16,
            )
            self.text_encoder.eval()

    # ------------------------------------------------------------------ #
    # Image helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resize_target(self, image: Image.Image) -> Image.Image:
        """Initial resize of the target image (BILINEAR + center_crop,
        16-aligned, ``area <= target_max_pixels``).

        This is the only resize the target image sees before VAE encode -
        the recipe does not run an additional ``auto_resize`` on targets.
        """
        return _image_crop_and_resize(
            image,
            max_pixels=self.config.target_max_pixels,
            height_division_factor=self.config.target_size_alignment,
            width_division_factor=self.config.target_size_alignment,
        )

    def _resize_for_vae(self, image: Image.Image) -> Image.Image:
        """Resize an edit image for VAE encoding (two-step).

        First the data-pipeline ``ImageCropAndResize`` (BILINEAR +
        center_crop, 16-aligned, ``area <= target_max_pixels``), then
        ``auto_resize`` to ``vae_image_area`` with ``vae_image_alignment``.
        Matches the upstream SFT recipe.
        """
        cropped = _image_crop_and_resize(
            image,
            max_pixels=self.config.target_max_pixels,
            height_division_factor=self.config.target_size_alignment,
            width_division_factor=self.config.target_size_alignment,
        )
        return _auto_resize_to_area(
            cropped,
            target_area=self.config.vae_image_area,
            alignment=self.config.vae_image_alignment,
        )

    def _resize_for_condition(self, image: Image.Image) -> Image.Image:
        """Resize an edit image for the text encoder's vision input.

        Three-step pipeline matching the SFT recipe's unit order
        (EditImageEmbedder runs first and overwrites ``inputs_shared["edit_image"]``
        with its auto-resized output before PromptEmbedder picks it up):

            1. ``_image_crop_and_resize`` (BILINEAR + center_crop, 16-aligned,
               area <= target_max_pixels).
            2. ``_auto_resize_to_area`` to ``vae_image_area`` with
               ``vae_image_alignment`` (the auto-resize EditImageEmbedder
               applies).
            3. ``_auto_resize_to_area`` to ``condition_image_area`` with
               ``condition_image_alignment`` (the ``resize_image(target_area=384^2)``
               step inside ``encode_prompt_edit_multi``).

        The intermediate (1024^2, 32-aligned) PIL is the source for the
        final 384^2 PIL resize - using a different source size silently
        changes the sub-pixel sampling and the resulting prompt embedding
        diverges from the reference cache.
        """
        cropped = _image_crop_and_resize(
            image,
            max_pixels=self.config.target_max_pixels,
            height_division_factor=self.config.target_size_alignment,
            width_division_factor=self.config.target_size_alignment,
        )
        vae_sized = _auto_resize_to_area(
            cropped,
            target_area=self.config.vae_image_area,
            alignment=self.config.vae_image_alignment,
        )
        return _auto_resize_to_area(
            vae_sized,
            target_area=self.config.condition_image_area,
            alignment=self.config.condition_image_alignment,
        )

    def _preprocess_image_tensor(self, image: Image.Image) -> torch.Tensor:
        """PIL.Image -> ``(1, 3, H, W)`` float tensor in ``[-1, +1]``.

        Linear rescale, no further normalisation. The cast to the VAE's
        dtype happens BEFORE the ``* (2/255) - 1`` rescale, so the
        rescale runs at the encode dtype (bf16 in training). Doing the
        rescale in fp32 and casting at the end is algebraically
        equivalent but leaves ~1 bf16 ULP of pixel-level noise that the
        deep VAE conv stack amplifies to ~6e-2 on the output latent;
        rescale-in-bf16 matches the reference cache bit-for-bit.
        """
        import numpy as np

        # PIL -> numpy (H, W, C) float32 -> cast to VAE dtype on VAE's
        # device -> rescale to [-1, +1] in that dtype -> (1, C, H, W).
        pixel = torch.from_numpy(np.array(image, dtype=np.float32))
        pixel = pixel.to(device=self._execution_device, dtype=self.vae.encoder.conv_in.weight.dtype)
        pixel = pixel * (2.0 / 255.0) - 1.0
        return pixel.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def _vae_encode(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode a single image to a normalised latent.

        Input:  ``(1, 3, H, W)`` in ``[-1, +1]``.
        Output: ``(1, 16, H/8, W/8)`` already mean/std-normalised by the
                upstream ``latents_mean`` / ``latents_std``.

        4-D output is contractually important - the cache format spec
        requires it and the DiT consumes a 4-D latent. Do not return a 5-D
        ``(1, 16, 1, H/8, W/8)`` tensor or the unnormalised 32-channel
        posterior.
        """
        return self.vae.encode(pixel)

    # ------------------------------------------------------------------ #
    # Text encoding                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """Split a batch of attention-masked hidden states into per-sample
        un-padded chunks.
        """
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompts: List[str],
        condition_images_per_sample: List[List[Image.Image]],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the prompt-edit text encoder (multi-image variant).

        For QIE-2511 each sample carries one or more edit images. They are
        prefixed by the ``Picture {n}:`` block in front of the user prompt;
        the user message itself is just ``{image_blocks}{prompt}``.

        We encode the batch one sample at a time so each prompt's vision
        tokens land in the right positions, then return the un-padded
        per-sample hidden states. The offline saver stores them as-is so
        downstream batching can re-pad to the actual batch max-length.

        Returns:
            (per_sample_embeds, per_sample_masks) - lists of length
            ``len(prompts)``. Each embed has shape
            ``(actual_seq_len, hidden_dim)``; each mask has shape
            ``(actual_seq_len,)``.
        """
        assert len(prompts) == len(condition_images_per_sample), "one image list per prompt"
        backbone = self.text_encoder.model
        device = next(backbone.parameters()).device
        dtype = next(backbone.parameters()).dtype

        per_sample_embeds: List[torch.Tensor] = []
        per_sample_masks: List[torch.Tensor] = []
        for prompt, cond_imgs in zip(prompts, condition_images_per_sample):
            base_img_prompt = "".join(_IMG_PROMPT_TEMPLATE.format(i + 1) for i in range(len(cond_imgs)))
            text = _PROMPT_TEMPLATE.format(base_img_prompt + prompt)
            model_inputs = self.processor(
                text=[text],
                images=cond_imgs if len(cond_imgs) > 0 else None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            hidden = self.text_encoder(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                image_grid_thw=model_inputs.get("image_grid_thw"),
            )[-1]
            split = self._extract_masked_hidden(hidden, model_inputs["attention_mask"])
            embeds = split[0][_PROMPT_DROP_IDX:]
            mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)
            per_sample_embeds.append(embeds.to(dtype=dtype).cpu())
            per_sample_masks.append(mask.cpu())
        return per_sample_embeds, per_sample_masks

    # ------------------------------------------------------------------ #
    # Public API consumed by the trainer / offline saver                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_condition(
        self,
        inputs: List[str],
        images: List[Any],
        videos: Optional[List[Any]] = None,
        outputs: Optional[List[Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the text encoder + VAE on a batch of samples.

        Args:
            inputs: per-sample prompt strings, ``len == batch_size``.
            images: per-sample dict ``{"image": [target_pil],
                "edit_image": [edit_pil, ...]}``. SFT recipe expects exactly
                one ``image`` and one ``edit_image`` per sample.

        Returns a dict with four lists, each of length ``batch_size``:
            * ``latents``:      ``(1, 16, H/8, W/8)`` target VAE latent
              (already normalised). 4-D.
            * ``edit_latents``: list of ``(1, 16, He/8, We/8)`` edit VAE
              latents per sample (one or more depending on edit image
              count). 4-D each.
            * ``prompt_emb``:   ``(L_actual, 3584)`` prompt embedding per
              sample, un-padded.
            * ``prompt_emb_mask``: ``(L_actual,)`` int64 mask per sample,
              all ones over the real tokens (the un-padded split has no
              padded positions).
        """
        del videos, outputs, kwargs

        assert isinstance(inputs, list), "inputs must be a list of prompts"
        prompts: List[str] = list(inputs)
        target_images: List[Image.Image] = []
        edit_images_per_sample: List[List[Image.Image]] = []
        for sample_images in images:
            assert isinstance(sample_images, dict), (
                "Each sample's 'images' entry must be a dict with 'image' and 'edit_image' keys."
            )
            target = sample_images.get("image")
            if isinstance(target, list):
                assert len(target) == 1, "Exactly one target image per sample is required."
                target = target[0]
            assert isinstance(target, Image.Image), "Target image must be a PIL.Image"
            target_images.append(target)

            edit = sample_images.get("edit_image") or []
            if isinstance(edit, Image.Image):
                edit = [edit]
            edit_images_per_sample.append(list(edit))

        # VAE: target image latents.
        target_latents: List[torch.Tensor] = []
        for img in target_images:
            resized = self._resize_target(img)
            pixel = self._preprocess_image_tensor(resized)
            target_latents.append(self._vae_encode(pixel).cpu())

        # VAE: edit-image latents (1024^2 area).
        # Cache the vae-sized PIL per ref so the text-encoder path can derive
        # the condition image from the exact same intermediate (single-step
        # ``_auto_resize_to_area`` below). This mirrors the reference SFT
        # data flow: the edit-image embedder writes the vae-sized PIL back
        # into ``inputs_shared["edit_image"]``, and the prompt embedder reads
        # it and single-step resizes to the condition area for the vision
        # tower. Decoupling the two resize chains - one from raw PIL for
        # VAE, another from raw PIL for the text encoder - lets per-step
        # rounding accumulate and produces vision-tower-token drift even
        # when the final PIL sizes match.
        edit_latents: List[List[torch.Tensor]] = []
        edit_vae_sized_per_sample: List[List[Image.Image]] = []
        for ref_imgs in edit_images_per_sample:
            sample_latents: List[torch.Tensor] = []
            sample_vae_sized: List[Image.Image] = []
            for ref in ref_imgs:
                vae_sized = self._resize_for_vae(ref)
                sample_vae_sized.append(vae_sized)
                pixel = self._preprocess_image_tensor(vae_sized)
                sample_latents.append(self._vae_encode(pixel).cpu())
            edit_latents.append(sample_latents)
            edit_vae_sized_per_sample.append(sample_vae_sized)

        # Text encoder: derive condition images from the cached vae-sized
        # PILs (single ``_auto_resize_to_area`` to ``condition_image_area``).
        # Multi-image template (``Picture {n}:`` per ref) is applied inside
        # ``_encode_prompt``.
        condition_images_per_sample: List[List[Image.Image]] = []
        for vae_sized_refs in edit_vae_sized_per_sample:
            assert len(vae_sized_refs) >= 1, "At least one edit image is required per sample."
            condition_images_per_sample.append(
                [
                    _auto_resize_to_area(
                        v,
                        target_area=self.config.condition_image_area,
                        alignment=self.config.condition_image_alignment,
                    )
                    for v in vae_sized_refs
                ]
            )
        prompt_embeds, prompt_masks = self._encode_prompt(prompts, condition_images_per_sample)

        return {
            "latents": target_latents,
            "edit_latents": edit_latents,
            "prompt_emb": prompt_embeds,
            "prompt_emb_mask": prompt_masks,
        }

    # ------------------------------------------------------------------ #
    # Training-time noise / timestep / target construction               #
    # ------------------------------------------------------------------ #

    def process_condition(
        self,
        latents: List[torch.Tensor],
        edit_latents: List[List[torch.Tensor]],
        prompt_emb: List[torch.Tensor],
        prompt_emb_mask: List[torch.Tensor],
        **unused,
    ) -> dict[str, Any]:
        """Turn clean cached per-sample outputs into a DiT.forward batch.

        On first call switches the scheduler into training mode (populates
        the bsmntw weights and the 1000-step timesteps array). Then samples
        a uniform timestep on ``[0, num_train_timesteps)`` and Gaussian
        noise from the per-rank generator, and builds:

            x_t   = (1 - sigma) * x + sigma * noise
            target = noise - x
            weight = bsmntw[timestep_id]

        The collator hands us length-1 lists (``micro_batch_size=1``); we
        unwrap them. Multi-sample batching is precluded by variable (H, W)
        per sample.
        """
        del unused

        if not self._timesteps_ready:
            self.scheduler.set_timesteps(
                num_inference_steps=self.config.num_train_timesteps,
                training=True,
            )
            self._timesteps_ready = True

        assert len(latents) == 1, "DiT SFT runs at micro_batch_size=1."
        x = latents[0]
        edit_list = list(edit_latents[0]) if edit_latents else []
        pe = prompt_emb[0]
        pm = prompt_emb_mask[0]
        device = x.device
        dtype = x.dtype

        # Per-rank deterministic sampling. ``randn_like(generator=)`` lands
        # in torch 2.10+; until then sample on the generator's device with
        # ``randn(shape, generator=)`` and move to the latent's device.
        n_steps = len(self.scheduler.timesteps)
        timestep_ids = torch.randint(
            0,
            n_steps,
            (x.shape[0],),
            device=self.generator.device,
            generator=self.generator,
        )
        timestep = self.scheduler.timesteps[timestep_ids.cpu()].to(device=device, dtype=dtype)
        noise = torch.randn(
            x.shape,
            device=self.generator.device,
            dtype=dtype,
            generator=self.generator,
        ).to(device)

        x_t = self.scheduler.add_noise(x, noise, timestep)
        training_target = self.scheduler.training_target(x, noise, timestep)
        training_weight = self.scheduler.training_weight(timestep).to(device=device, dtype=dtype)

        return {
            "latents": x_t,
            "timestep": timestep,
            "prompt_emb": pe,
            "prompt_emb_mask": pm,
            "edit_latents": edit_list,
            "training_target": training_target,
            "training_weight": training_weight,
        }
