# Solution -- 08.01 requires_grad and backward
import torch


def gradient_of_quadratic(values: list[float]) -> torch.Tensor:
    """d/dx of sum(3 x^2 + 2 x), evaluated at `values`, by autograd."""
    x = torch.tensor(values, requires_grad=True)
    f = (3 * x**2 + 2 * x).sum()
    f.backward()
    return x.grad


def mse_gradients(
    w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of mean((w x + b - y)^2) with respect to w and b."""
    w = w.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()
    return w.grad, b.grad


def test_gradient_of_quadratic_matches_calculus():
    g = gradient_of_quadratic([0.0, 1.0, -2.0])
    assert g is not None
    torch.testing.assert_close(g, torch.tensor([2.0, 8.0, -10.0]))


def test_mse_gradients_match_the_closed_form():
    torch.manual_seed(0)
    x = torch.randn(50)
    y = 2.0 * x + 1.0
    w = torch.tensor(0.5)
    b = torch.tensor(-0.5)
    gw, gb = mse_gradients(w, b, x, y)
    residual = w * x + b - y
    torch.testing.assert_close(gw, (2 * residual * x).mean())
    torch.testing.assert_close(gb, (2 * residual).mean())


def test_inputs_are_left_alone():
    w = torch.tensor(1.0)
    b = torch.tensor(0.0)
    mse_gradients(w, b, torch.ones(3), torch.ones(3))
    assert not w.requires_grad
    assert w.grad is None
