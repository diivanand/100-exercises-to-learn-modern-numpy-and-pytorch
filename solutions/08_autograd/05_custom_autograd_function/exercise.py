# Solution -- 08.05 A custom autograd Function
import torch


class StableSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.logaddexp(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return StableSoftplus.apply(x)


def test_forward_matches_the_library_and_is_stable():
    x = torch.tensor([-100.0, 0.0, 100.0])
    torch.testing.assert_close(softplus(x), torch.nn.functional.softplus(x))
    assert torch.isfinite(softplus(x)).all()


def test_backward_matches_the_library():
    x = torch.linspace(-4, 4, 9, requires_grad=True)
    softplus(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.sigmoid(x.detach()))


def test_gradcheck_in_float64():
    x = torch.randn(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(softplus, (x,))
