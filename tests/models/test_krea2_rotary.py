import pytest
import torch

import veomni.models.diffusers.krea2.krea2_transformer.modeling_krea2_transformer as krea2
from veomni.models.diffusers.krea2.krea2_transformer.modeling_krea2_transformer import (
    Krea2RotaryPosEmbed,
    _apply_rotary_emb,
    _get_1d_rotary_pos_embed,
)


def _reference_apply_rotary_emb(
    x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs_cis
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)
    cos = cos.repeat_interleave(2, dim=-1, output_size=cos.shape[-1] * 2)[None, :, None, :]
    sin = sin.repeat_interleave(2, dim=-1, output_size=sin.shape[-1] * 2)[None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _make_freqs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Krea2RotaryPosEmbed(theta=1000.0, axes_dim=[2, 2, 4])
    position_ids = torch.tensor(
        [[0, 0, 0], [1, 0, 1], [2, 3, 1], [3, 1, 4], [4, 2, 2], [5, 4, 3], [6, 2, 5]],
        device=device,
    )
    return rotary(position_ids)


def _reference_rotary_freqs(
    position_ids: torch.Tensor,
    theta: float = 1000.0,
    axes_dim: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if axes_dim is None:
        axes_dim = [2, 2, 4]
    cos_out = []
    sin_out = []
    freqs_dtype = torch.float64 if position_ids.device.type not in ("mps", "npu") else torch.float32
    pos = position_ids.float()
    for axis_idx, axis_dim in enumerate(axes_dim):
        cos, sin = _get_1d_rotary_pos_embed(
            axis_dim,
            pos[:, axis_idx],
            theta=theta,
            freqs_dtype=freqs_dtype,
        )
        cos_out.append(cos)
        sin_out.append(sin)
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def test_krea2_rotary_freq_cache_matches_reference_cpu():
    position_ids = torch.tensor(
        [[0, 0, 0], [1, 0, 1], [2, 3, 1], [3, 1, 4], [4, 2, 2], [5, 4, 3], [6, 2, 5]]
    )
    rotary = Krea2RotaryPosEmbed(theta=1000.0, axes_dim=[2, 2, 4])

    actual = rotary(position_ids)
    expected = _reference_rotary_freqs(position_ids)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_krea2_rotary_float_positions_use_eager_path_cpu():
    position_ids = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 1.0], [2.0, 3.25, 1.0], [3.0, 1.0, 4.5]]
    )
    rotary = Krea2RotaryPosEmbed(theta=1000.0, axes_dim=[2, 2, 4])

    actual = rotary(position_ids)
    expected = _reference_rotary_freqs(position_ids)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_krea2_rotary_eager_matches_reference_cpu():
    torch.manual_seed(0)
    x = torch.randn(2, 7, 3, 8, dtype=torch.float32)
    freqs_cis = _make_freqs(x.device)

    actual = _apply_rotary_emb(x, freqs_cis)
    expected = _reference_apply_rotary_emb(x, freqs_cis)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_krea2_rotary_custom_op_is_visible_to_sac_when_available():
    if krea2.rotary_interleaved_fwd is None:
        pytest.skip("Triton rotary custom op is unavailable")

    from veomni.distributed import sac

    rotary_ops = sac._collect_rotary_ops()
    assert torch.ops.veomni.rotary_interleaved_fwd.default in rotary_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused RoPE kernel")
def test_krea2_rotary_fused_matches_reference_cuda_forward_backward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    freqs_cis = _make_freqs(device)

    for dtype, rtol, atol in [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-2, 2e-2)]:
        x = torch.randn(2, 7, 3, 8, device=device, dtype=dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        actual = _apply_rotary_emb(x, freqs_cis)
        expected = _reference_apply_rotary_emb(x_ref, freqs_cis)

        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)

        torch.testing.assert_close(x.grad, x_ref.grad, rtol=rtol, atol=atol)
