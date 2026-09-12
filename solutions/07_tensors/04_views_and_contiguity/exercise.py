# Solution -- 07.04 Views and contiguity
import torch


def flatten_rows(m: torch.Tensor) -> torch.Tensor:
    """The elements of `m` in row-major order, as a 1-d tensor.

    A view when the layout allows it, a copy otherwise.
    """
    return m.reshape(-1)


def same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if `a` and `b` are views over the same memory."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def strides_of(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(t.stride())


def test_flatten_contiguous_is_a_view():
    m = torch.arange(12.0).reshape(3, 4)
    flat = flatten_rows(m)
    torch.testing.assert_close(flat, torch.arange(12.0))
    assert same_storage(flat, m)
    flat[0] = -1.0
    assert m[0, 0] == -1.0


def test_flatten_transposed_gives_row_major_of_the_transpose():
    m = torch.arange(6.0).reshape(2, 3)
    t = m.t()
    assert not t.is_contiguous()
    assert strides_of(t) == (1, 3)
    flat = flatten_rows(t)
    torch.testing.assert_close(flat, torch.tensor([0.0, 3.0, 1.0, 4.0, 2.0, 5.0]))
    # This one had to be a copy.
    assert not same_storage(flat, m)


def test_same_storage_sees_through_views():
    a = torch.zeros(4, 4)
    assert same_storage(a, a[1:3])
    assert same_storage(a, a.t())
    assert not same_storage(a, a.clone())
