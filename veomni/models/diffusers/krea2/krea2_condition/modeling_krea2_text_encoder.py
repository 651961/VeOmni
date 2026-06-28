from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)


@dataclass
class TextEncoderConfig:
    model_id: str
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Krea2Qwen3VLConditioner(torch.nn.Module):
    """Qwen3-VL conditioner used by Krea-2 image editing."""

    def __init__(
        self,
        version: str,
        tokenizer_path: str | None = None,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(version)
        architectures = config.architectures or []
        if "Qwen3VLForConditionalGeneration" in architectures:
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(version)
        else:
            self.qwen = Qwen3VLModel.from_pretrained(version)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or version, max_length=max_length)
        self.processor = self._build_processor(version, self.tokenizer, max_length=max_length)
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5
        self.edit_prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
            "objects, background), then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the "
            "original input where appropriate.<|im_end|>\n<|im_start|>user\n"
        )
        self.vision_start = "<|vision_start|>"

    @staticmethod
    def _build_processor(version: str, tokenizer, max_length: int):
        try:
            return AutoProcessor.from_pretrained(version, max_length=max_length)
        except (OSError, ValueError):
            image_processor = Qwen2VLImageProcessor(
                size={"longest_edge": 16777216, "shortest_edge": 65536},
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            video_processor = Qwen3VLVideoProcessor(
                size={"longest_edge": 25165824, "shortest_edge": 4096},
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            return Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                chat_template=getattr(tokenizer, "chat_template", None),
            )

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.tokenizer(text=suffix_text, return_tensors="pt").to(
            self.qwen.device, non_blocking=True
        )
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]
            return hiddens, mask

    def forward_edit(self, text: list[str], images: list) -> tuple[Tensor, Tensor]:
        system_prompt = self.edit_prompt_template_encode_prefix.removeprefix(
            "<|im_start|>system\n"
        ).removesuffix("<|im_end|>\n<|im_start|>user\n")
        first_image_label_ids = self.tokenizer("Picture 1: ", add_special_tokens=False)["input_ids"]
        conversations = []
        for item in text:
            user_content = []
            for i, image in enumerate(images):
                user_content.append({"type": "text", "text": f"Picture {i + 1}: "})
                user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": item})
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ]
            )

        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                padding=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.qwen.device, non_blocking=True)
            mask = inputs["attention_mask"].bool()
            inputs["attention_mask"] = mask
            states = self.qwen(**inputs, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            prefix_idx = 0
            vision_start_id = self.tokenizer.convert_tokens_to_ids(self.vision_start)
            for ids, sample_mask in zip(inputs["input_ids"], mask):
                valid_ids = ids[sample_mask]
                if first_image_label_ids:
                    label_len = len(first_image_label_ids)
                    for idx in range(0, valid_ids.numel() - label_len + 1):
                        if valid_ids[idx : idx + label_len].tolist() == first_image_label_ids:
                            prefix_idx = idx
                            break
                    if prefix_idx > 0:
                        break
                vision_start_positions = (valid_ids == vision_start_id).nonzero(as_tuple=True)[0]
                image_positions = (valid_ids == self.qwen.config.image_token_id).nonzero(as_tuple=True)[0]
                if len(vision_start_positions) > 0:
                    prefix_idx = int(vision_start_positions[0])
                    break
                if len(image_positions) > 0:
                    prefix_idx = int(image_positions[0])
                    break
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]
            return hiddens, mask
