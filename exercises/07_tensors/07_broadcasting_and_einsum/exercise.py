# =============================================================================
#  07.07 -- Broadcasting and einsum
# =============================================================================
#
#  Broadcasting works exactly as in NumPy (02.01): shapes are aligned from
#  the right, a 1 stretches, a missing dimension counts as 1. The failure
#  modes are the same too, and the silent one is worse than the loud one.
#
#  Scaling each ROW of an (n, d) matrix by a length-n vector needs the
#  vector as a column, `s[:, None]`. Written as `x * s` it either raises
#  (when n != d: "The size of tensor a must match the size of tensor b at
#  non-singleton dimension 1") or, when n == d, scales the COLUMNS instead
#  and nothing complains. A test with a square matrix is the only way to
#  catch the second case, which is why there is one below.
#
#  `torch.einsum` writes the contraction in index notation and is often the
#  clearest way to say what you mean:
#
#      torch.einsum("bi,bi->b", a, b)     a batch of dot products
#      torch.einsum("bi,bj->bij", a, b)   a batch of outer products
#      torch.einsum("bij,bjk->bik", a, b) the same as torch.bmm(a, b), or a @ b
#
#  `@` on tensors with more than two dimensions is batched matmul with
#  broadcasting over the leading dimensions, so `(B, n, k) @ (k, m)` works
#  and `(B, n, k) @ (B, k, m)` works.
#
#  TASK
#    Fix `scale_rows`, and write `batched_dot` and `batched_outer` with
#    einsum.
#
#  RUN IT
#    ./npt test 07_07
#
# =============================================================================

import torch


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Row i of the result is row i of `x` times s[i]."""
    # TODO: `s` broadcasts against the LAST dimension, the columns. Give it
    # a trailing axis so it lines up with the rows.
    return x * s


def batched_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, d), (B, d) -> (B,): the dot product of each pair of rows."""
    # TODO: one einsum, no loop.
    return torch.stack([torch.dot(a[i], b[i]) for i in range(a.shape[0])])


def batched_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, n), (B, m) -> (B, n, m): the outer product of each pair of rows."""
    # TODO: one einsum, no loop.
    return torch.stack([torch.outer(a[i], b[i]) for i in range(a.shape[0])])


def test_scale_rows_rectangular():
    x = torch.ones(3, 2)
    out = scale_rows(x, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(out, torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))


def test_scale_rows_square_is_not_scale_columns():
    # With n == d the wrong broadcast does not raise; it scales the columns.
    x = torch.ones(2, 2)
    out = scale_rows(x, torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(out, torch.tensor([[1.0, 1.0], [10.0, 10.0]]))


def test_batched_dot_and_outer():
    torch.manual_seed(0)
    a = torch.randn(5, 3)
    b = torch.randn(5, 4)
    dots = batched_dot(a, a)
    torch.testing.assert_close(dots, (a * a).sum(dim=1))
    outer = batched_outer(a, b)
    assert outer.shape == (5, 3, 4)
    torch.testing.assert_close(outer[2], torch.outer(a[2], b[2]))


def test_no_python_loops():
    import inspect

    for fn in (batched_dot, batched_outer):
        assert "for " not in inspect.getsource(fn)
