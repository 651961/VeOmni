"""DiT examples must reach the native parser and shared data transform."""

import sys
from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import parse_args
from veomni.models.auto import build_config
from veomni.trainer.dit_trainer import VeOmniDiTArguments


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["wan2.2_I2V_5B", "wan2.2_I2V_14B"])
def test_wan_i2v_yaml_uses_declared_arguments(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    # Kernel availability is independent of the example's dataclass contract.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dit.py",
            str(ROOT / "configs" / "dit" / f"{name}.yaml"),
            "--model.ops_implementation.load_balancing_loss_implementation",
            "eager",
        ],
    )
    args = parse_args(VeOmniDiTArguments)
    assert Path(args.model.config_path).is_file()
    config = build_config(args.model.config_path)
    assert config.model_type == "WanTransformer3DModel"
    assert args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.model.accelerator.fsdp_config.mixed_precision.param_dtype == "bfloat16"
    assert not args.model.accelerator.offload_config.enable_async_activation
    assert not args.model.accelerator.offload_config.enable_activation
    assert args.model.accelerator.gradient_checkpointing.enable
    assert args.train.training_task == "online_training"


@pytest.mark.parametrize(
    "source_hw,expected_hw,max_pixels",
    [
        ((1344, 768), (1344, 768), 1032192),
        ((768, 1344), (768, 1344), 1032192),
        ((864, 1184), (864, 1184), 1032192),
        ((2688, 1536), (1344, 768), 1032192),
        ((1344, 768), (480, 832), None),
    ],
)
def test_h3_video_resize_preserves_frame_and_keyframe_geometry(monkeypatch, source_hw, expected_hw, max_pixels):
    import imageio
    import numpy as np
    import torch

    from veomni.data.multimodal.dit.data_transform import process_minimax_h3_online_example

    h, w = source_hw
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[: h // 8] = [255, 0, 0]
    frame[-h // 8 :] = [0, 255, 0]
    frame[:, : w // 8] = [0, 0, 255]
    frame[:, -w // 8 :] = [255, 255, 255]

    class Reader:
        closed = False

        def get_meta_data(self):
            return {"fps": 24, "duration": 5 / 24}

        def iter_data(self):
            return iter([frame] * 4)  # Last requested frame exercises stream-end clamping.

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(imageio, "get_reader", lambda _: reader)
    options = {"video_max_pixels": max_pixels} if max_pixels is not None else {}
    sample = process_minimax_h3_online_example(
        {"video": "unused.mp4", "prompt": "dance\nwith music"},
        "minimax_h3",
        min_frames=5,
        keyframe_indices=[0],
        **options,
    )[0]
    assert reader.closed and len(sample["videos"]) == 5
    assert sample["inputs"] == "dance\nwith music" and sample["audios"] is None
    assert all(tuple(video.shape) == (3, *expected_hw) for video in sample["videos"])
    keyframe = np.array(sample["images"][0])
    torch.testing.assert_close(sample["videos"][0], torch.from_numpy(keyframe).permute(2, 0, 1).float() / 255)
    if max_pixels is not None:
        out_h, out_w = expected_hw
        assert out_h % 32 == out_w % 32 == 0 and out_h * out_w <= max_pixels
        for y, x in [(0, w // 2), (h - 1, w // 2), (h // 2, 0), (h // 2, w - 1)]:
            np.testing.assert_array_equal(
                keyframe[y * (out_h - 1) // (h - 1), x * (out_w - 1) // (w - 1)], frame[y, x]
            )
