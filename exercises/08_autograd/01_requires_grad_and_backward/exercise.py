# =============================================================================
#  08.01 -- requires_grad and backward
# =============================================================================
#
#  Autograd is the reason PyTorch exists. Mark a tensor with
#  `requires_grad=True`, compute a scalar from it, call `.backward()` on the
#  scalar, and every marked tensor upstream gets a `.grad` holding the
#  partial derivative of that scalar with respect to it. Godoy, ch. 1
#  "Autograd", calls it "PyTorch's automatic differentiation package", and
#  the whole of his chapter 1 is replacing hand-derived gradients with it.
#
#  Three things to know now, and one per exercise after this:
#
#   * Only a tensor created with `requires_grad=True` (a LEAF) accumulates a
#     `.grad`. A tensor you forgot to mark has `.grad` of None, and there is
#     no error: the gradient was never asked for.
#   * `backward()` needs a SCALAR. For a vector output you either reduce it
#     first (`loss.mean()`) or pass a vector to `backward` (08.08).
#   * The chain rule is the whole algorithm (Kneusel, MfDL ch. 10
#     "Backpropagation"; Volpe et al., ch. 1 "Mathematical Derivation"): for
#     f(x) = sum(3x^2 + 2x), df/dx = 6x + 2, and autograd will agree to the
#     last bit.
#
#  TASK
#    Make `gradient_of_quadratic` return the gradient autograd computes, and
#    `mse_gradients` return the gradients with respect to w and b.
#
#  RUN IT
#    ./npt test 08_01
#
# =============================================================================

import torch


def gradient_of_quadratic(values: list[float]) -> torch.Tensor:
    """d/dx of sum(3 x^2 + 2 x), evaluated at `values`, by autograd."""
    # TODO: nothing here asks for a gradient, so x.grad stays None.
    x = torch.tensor(values)
    f = (3 * x**2 + 2 * x).sum()
    return x.grad


def mse_gradients(
    w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of mean((w x + b - y)^2) with respect to w and b."""
    # TODO: the loss is a vector, and backward() needs a scalar. Reduce it.
    # (You also need w and b to require grad; the tests pass plain tensors,
    # so make grad-tracking copies here with .detach().requires_grad_().)
    loss = (w * x + b - y) ** 2
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
