# Solution -- 09.01 Linear regression by hand
import torch


def make_data(n: int = 100, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Points on the line y = 2x + 1, plus a little Gaussian noise."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=generator)
    noise = 0.1 * torch.randn(n, 1, generator=generator)
    y = 1.0 + 2.0 * x + noise
    return x, y


def train(
    x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit y = b + w * x by gradient descent and return (w, b)."""
    generator = torch.Generator().manual_seed(0)
    w = torch.randn(1, generator=generator, requires_grad=True)
    b = torch.randn(1, generator=generator, requires_grad=True)

    for _ in range(steps):
        yhat = b + w * x
        loss = ((yhat - y) ** 2).mean()
        loss.backward()

        # The update is not part of the model: it must not be recorded in the
        # graph, and it must modify w and b IN PLACE so they stay the leaf
        # tensors autograd is tracking. `w = w - lr * w.grad` would rebind the
        # name to a brand-new tensor with no gradient history.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # Gradients accumulate; clear them before the next backward pass.
        w.grad.zero_()
        b.grad.zero_()

    return w, b


def test_recovers_the_true_line():
    x, y = make_data()
    w, b = train(x, y)
    assert abs(w.item() - 2.0) < 0.1
    assert abs(b.item() - 1.0) < 0.1


def test_parameters_stay_leaves():
    # A parameter that was rebound instead of updated in place is no longer a
    # leaf, has no gradient history, and cannot be trained further.
    x, y = make_data(n=20)
    w, b = train(x, y, steps=5)
    assert w.is_leaf and b.is_leaf
    assert w.requires_grad and b.requires_grad
    assert w.grad is not None and float(w.grad.abs().sum()) == 0.0


def test_loss_decreases_with_more_steps():
    x, y = make_data()

    def mse(steps: int) -> float:
        w, b = train(x, y, steps=steps)
        with torch.no_grad():
            return float(((b + w * x - y) ** 2).mean())

    assert mse(200) < mse(20) < mse(2)
