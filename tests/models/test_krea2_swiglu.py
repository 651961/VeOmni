import pytest
import torch

import veomni.models.diffusers.krea2.krea2_transformer.modeling_krea2_transformer as krea2
from veomni.models.diffusers.krea2.krea2_transformer.modeling_krea2_transformer import Krea2SwiGLU


def _run_swiglu(module: Krea2SwiGLU, x: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    out = module(x)
    out.backward(grad)
    param_grads = {name: param.grad.detach().clone() for name, param in module.named_parameters()}
    return out.detach(), x.grad.detach().clone(), param_grads


def test_krea2_swiglu_eager_matches_reference_cpu():
    torch.manual_seed(0)
    module = Krea2SwiGLU(dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 8, requires_grad=True)
    grad = torch.randn(2, 5, 8)

    out = module(x)
    expected = module.down(torch.nn.functional.silu(module.gate(x)) * module.up(x))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    out.backward(grad)
    expected.backward(grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused SwiGLU kernel")
def test_krea2_swiglu_fused_matches_eager_cuda_forward_backward():
    torch.manual_seed(0)
    original_liger_silu_mul = krea2.LigerSiLUMulFunction
    assert original_liger_silu_mul is not None

    for dtype, rtol, atol in [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-2, 2e-2)]:
        eager = Krea2SwiGLU(dim=8, hidden_dim=32).cuda().to(dtype)
        fused = Krea2SwiGLU(dim=8, hidden_dim=32).cuda().to(dtype)
        fused.load_state_dict(eager.state_dict())

        x = torch.randn(2, 5, 8, device="cuda", dtype=dtype, requires_grad=True)
        x_eager = x.detach().clone().requires_grad_(True)
        x_fused = x.detach().clone().requires_grad_(True)
        grad = torch.randn(2, 5, 8, device="cuda", dtype=dtype)

        try:
            krea2.LigerSiLUMulFunction = None
            out_eager, x_grad_eager, param_grads_eager = _run_swiglu(eager, x_eager, grad)

            krea2.LigerSiLUMulFunction = original_liger_silu_mul
            out_fused, x_grad_fused, param_grads_fused = _run_swiglu(fused, x_fused, grad)
        finally:
            krea2.LigerSiLUMulFunction = original_liger_silu_mul

        torch.testing.assert_close(out_fused, out_eager, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_grad_fused, x_grad_eager, rtol=rtol, atol=atol)
        for name in param_grads_eager:
            torch.testing.assert_close(param_grads_fused[name], param_grads_eager[name], rtol=rtol, atol=atol)
