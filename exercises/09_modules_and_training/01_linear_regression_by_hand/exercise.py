# =============================================================================
#  09.01 -- Linear regression by hand
# =============================================================================
#
#  Before nn.Module, before optimisers, there are five steps, and every
#  training loop you will ever write is these five steps in a costume
#  (Godoy, *Deep Learning with PyTorch Step-by-Step*, ch. 1, "Gradient
#  Descent", steps 0 to 5; Kneusel, *Math for Deep Learning*, ch. 11):
#
#      0. start the parameters somewhere random;
#      1. compute the model's predictions           yhat = b + w * x
#      2. compute the loss                          mean((yhat - y)^2)
#      3. compute the gradients                     loss.backward()
#      4. update the parameters                     w -= lr * w.grad
#      5. repeat.
#
#  Step 3 is the one PyTorch does for you. A tensor created with
#  `requires_grad=True` is a LEAF of the computation graph; every operation
#  on it is recorded, and `loss.backward()` walks the record backwards and
#  leaves d(loss)/d(w) in `w.grad` (chapter 08).
#
#  Step 4 is the one people get wrong, in two ways:
#
#   * The update must happen outside autograd's view, inside `torch.no_grad()`.
#     Otherwise the subtraction is itself recorded, and the graph grows by one
#     node per step, for ever.
#
#   * The update must be IN PLACE: `w -= ...`, not `w = w - ...`. The second
#     form binds the name `w` to a new tensor -- inside no_grad, one with no
#     gradient history at all. On the next step the loss no longer depends on
#     anything that requires a gradient, and `backward()` raises. Godoy shows
#     this exact failure in ch. 1, "Updating Parameters".
#
#  And the small one: gradients ACCUMULATE. `backward()` adds to `.grad`; it
#  does not overwrite it. Zero them after each update (ch. 1, "zero_").
#
#  TASK
#    Make `train` update its parameters in place, and clear the gradients
#    between steps.
#
#  RUN IT
#    ./npt test 09_01
#
# =============================================================================

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

        # TODO: this rebinds `w` and `b` to fresh tensors that autograd knows
        # nothing about. Update them in place instead, and zero the gradients
        # afterwards.
        with torch.no_grad():
            w = w - lr * w.grad
            b = b - lr * b.grad

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
