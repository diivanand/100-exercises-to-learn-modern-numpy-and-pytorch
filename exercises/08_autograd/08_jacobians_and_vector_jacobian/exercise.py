# =============================================================================
#  08.08 -- Jacobians and vector-Jacobian products
# =============================================================================
#
#  For a function from R^n to R^m the derivative is the m x n Jacobian
#  (Kneusel, MfDL ch. 8 "Jacobians and Hessians"). Reverse-mode autograd
#  never builds it: what `backward` computes is a VECTOR-JACOBIAN PRODUCT,
#  v^T J, for a vector v you supply. For a scalar loss, v is the number 1
#  and v^T J is the gradient -- which is why `backward()` on a scalar needs
#  no argument, and on a non-scalar raises "grad can be implicitly created
#  only for scalar outputs". Pass the vector: `y.backward(gradient=v)`.
#
#  When you do want the whole matrix, `torch.func.jacrev(f)(x)` (reverse
#  mode, one vjp per output row) or `torch.func.jacfwd` (forward mode, one
#  jvp per input column) build it for you; pick reverse when m < n and
#  forward when m > n.
#
#  TASK
#    Fix `vjp` to pass the vector, and write `jacobian` with torch.func.
#
#  RUN IT
#    ./npt test 08_08
#
# =============================================================================

from collections.abc import Callable

import torch


def vjp(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """v^T J_f(x), computed by one backward pass."""
    x = x.detach().requires_grad_(True)
    y = f(x)
    # TODO: y is a vector; backward() needs to be told which vector to
    # multiply the Jacobian by.
    y.backward()
    return x.grad


def jacobian(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The full (m, n) Jacobian of f at x."""
    # TODO: one row per output, one backward pass each, stacked -- or let
    # torch.func.jacrev do exactly that.
    return vjp(f, x, torch.ones(f(x).shape[0]))


A = torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])


def linear(x: torch.Tensor) -> torch.Tensor:
    return A @ x


def test_vjp_of_a_linear_map():
    x = torch.tensor([1.0, 1.0, 1.0])
    v = torch.tensor([2.0, 5.0])
    torch.testing.assert_close(vjp(linear, x, v), v @ A)


def test_jacobian_of_a_linear_map_is_the_matrix():
    x = torch.tensor([0.5, -0.5, 2.0])
    j = jacobian(linear, x)
    assert j.shape == (2, 3)
    torch.testing.assert_close(j, A)


def test_jacobian_of_an_elementwise_function_is_diagonal():
    x = torch.tensor([0.0, 1.0, 2.0])
    j = jacobian(torch.exp, x)
    torch.testing.assert_close(j, torch.diag(torch.exp(x)))
