# =============================================================================
#  07.04 -- Views and contiguity
# =============================================================================
#
#  A tensor is a view over storage plus a shape and STRIDES: how many
#  elements to step in memory for one step along each dimension. A
#  contiguous (4, 3) float tensor has strides (3, 1). Its transpose is the
#  same storage with shape (3, 4) and strides (1, 3) -- free, no copy, and
#  `is_contiguous()` is False.
#
#  `view(shape)` reinterprets the SAME storage and therefore requires the
#  memory to be laid out compatibly; on a transposed tensor it raises
#  "view size is not compatible with input tensor's size and stride".
#  `reshape(shape)` returns a view when it can and a copy when it must, so
#  it always works -- and you no longer know whether writes go through.
#  `.contiguous()` copies only if needed and gives you a tensor you CAN
#  view.
#
#  The same story as NumPy (03.01, 03.02); PyTorch just has two verbs where
#  NumPy has one. Godoy, ch. 1 "Tensor", shows `view()` sharing memory with
#  its source and recommends `clone().detach()` for an independent copy.
#
#  TASK
#    Make `flatten_rows` work on any layout, and make `same_storage`
#    honest.
#
#  RUN IT
#    ./npt test 07_04
#
# =============================================================================

import torch


def flatten_rows(m: torch.Tensor) -> torch.Tensor:
    """The elements of `m` in row-major order, as a 1-d tensor.

    A view when the layout allows it, a copy otherwise.
    """
    # TODO: view() demands a compatible memory layout and raises on a
    # transposed tensor. Use the verb that copies only when it has to.
    return m.view(-1)


def same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if `a` and `b` are views over the same memory."""
    # TODO: `is` compares Python objects; two views are two objects.
    # Compare the storage's data pointer instead.
    return a is b


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
