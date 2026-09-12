# Solution -- 08.08 Jacobians and vector-Jacobian products
from collections.abc import Callable

import torch


def vjp(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """v^T J_f(x), computed by one backward pass."""
    x = x.detach().requires_grad_(True)
    y = f(x)
    y.backward(gradient=v)
    return x.grad


def jacobian(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The full (m, n) Jacobian of f at x."""
    return torch.func.jacrev(f)(x)


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
