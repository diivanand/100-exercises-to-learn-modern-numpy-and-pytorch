# Solution -- 07.07 Broadcasting and einsum
import torch


def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Row i of the result is row i of `x` times s[i]."""
    return x * s[:, None]


def batched_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, d), (B, d) -> (B,): the dot product of each pair of rows."""
    return torch.einsum("bi,bi->b", a, b)


def batched_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, n), (B, m) -> (B, n, m): the outer product of each pair of rows."""
    return torch.einsum("bi,bj->bij", a, b)


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
