# =============================================================================
#  08.06 -- Higher-order gradients
# =============================================================================
#
#  `backward()` is the convenient interface; `torch.autograd.grad(outputs,
#  inputs)` is the functional one. It RETURNS the gradients instead of
#  accumulating them into `.grad`, which makes it the right tool when the
#  gradient is an intermediate result you want to keep computing with.
#
#  Differentiating twice needs the first gradient to itself be part of the
#  graph: `torch.autograd.grad(..., create_graph=True)`. Without it the
#  returned gradient is a plain tensor with no grad_fn, and the second call
#  raises "One of the differentiated Tensors does not require grad" (or
#  returns None with `allow_unused`). Second derivatives are what gradient
#  penalties, curvature estimates and meta-learning need; Kneusel, MfDL
#  ch. 8 "Jacobians and Hessians", is the mathematics.
#
#  `torch.autograd.gradgradcheck` is gradcheck for the second derivative.
#
#  TASK
#    Make `second_derivative` compute f''(x), and `hessian_diagonal` the
#    diagonal of the Hessian of a scalar function of a vector.
#
#  RUN IT
#    ./npt test 08_06
#
# =============================================================================

from collections.abc import Callable

import torch


def second_derivative(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """f''(x) for a scalar function f, elementwise over `x`."""
    x = x.detach().requires_grad_(True)
    y = f(x).sum()
    # TODO: without create_graph the first gradient is not differentiable,
    # and the second call has nothing to differentiate.
    (dy,) = torch.autograd.grad(y, x)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    return d2y


def hessian_diagonal(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """d^2 f / dx_i^2 for a scalar-valued f of a vector x."""
    x = x.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(x), x, create_graph=True)
    # TODO: each grad[i] must be differentiated with respect to x again,
    # keeping the graph alive between iterations, and the i-th entry of the
    # i-th row taken.
    return grad


def test_second_derivative_of_a_cubic():
    x = torch.tensor([0.0, 1.0, 2.0])
    torch.testing.assert_close(second_derivative(lambda t: t**3, x), 6 * x)


def test_second_derivative_of_sin():
    x = torch.linspace(0, 3, 7)
    torch.testing.assert_close(second_derivative(torch.sin, x), -torch.sin(x))


def test_hessian_diagonal_of_a_quadratic_form():
    a = torch.tensor([[2.0, 1.0], [1.0, 4.0]])
    x = torch.tensor([0.5, -1.0])
    diag = hessian_diagonal(lambda v: 0.5 * v @ a @ v, x)
    torch.testing.assert_close(diag, torch.tensor([2.0, 4.0]))


def test_hessian_diagonal_against_the_full_hessian():
    x = torch.tensor([0.3, 0.7, -1.2])

    def f(v: torch.Tensor) -> torch.Tensor:
        return (v**3).sum() + torch.exp(v).prod()

    full = torch.autograd.functional.hessian(f, x)
    torch.testing.assert_close(hessian_diagonal(f, x), torch.diagonal(full))
