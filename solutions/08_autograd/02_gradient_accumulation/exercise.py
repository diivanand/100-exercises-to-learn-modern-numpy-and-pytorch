# Solution -- 08.02 Gradients accumulate
import torch


def sgd(x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> float:
    """Fit y = w x by plain gradient descent, returning w."""
    w = torch.zeros((), requires_grad=True)
    for _ in range(steps):
        loss = ((w * x - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad = None
    return w.item()


def _reference(x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> float:
    # The closed-form gradient, recomputed from scratch each step.
    w = 0.0
    for _ in range(steps):
        grad = (2 * (w * x - y) * x).mean().item()
        w -= lr * grad
    return w


def test_sgd_converges():
    torch.manual_seed(0)
    x = torch.randn(64)
    y = 3.0 * x
    assert abs(sgd(x, y, lr=0.1, steps=100) - 3.0) < 1e-3


def test_sgd_matches_a_fresh_gradient_each_step():
    torch.manual_seed(1)
    x = torch.randn(64)
    y = -1.5 * x
    for steps in (1, 2, 5):
        got = sgd(x, y, lr=0.05, steps=steps)
        want = _reference(x, y, lr=0.05, steps=steps)
        assert abs(got - want) < 1e-5, f"after {steps} steps: {got} vs {want}"
