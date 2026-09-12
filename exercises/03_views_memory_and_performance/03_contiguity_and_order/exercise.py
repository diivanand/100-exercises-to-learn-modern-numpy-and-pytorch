# =============================================================================
#  03.03 -- Contiguity and memory order
# =============================================================================
#
#  A 2-D array is one block of memory. Row-major ("C order") lays row 0 out
#  first, then row 1; column-major ("F order", for Fortran) lays column 0
#  out first. NumPy defaults to C order and can hold either, and the strides
#  (03.02) record which. An array whose elements sit in one unbroken
#  row-major run is "C-contiguous"; `a.flags` tells you:
#
#      a.flags.c_contiguous     a.flags.f_contiguous
#
#  Johansson, Numerical Python 3rd ed., ch. 2 "Order of Array Data in
#  Memory" explains the two layouts and where they come from.
#
#  It matters in two places. Any code that expects a plain C buffer -- a C
#  extension, PyTorch's `from_numpy` for some dtypes, an image encoder,
#  `np.frombuffer` -- needs C-contiguous input, and `a.T` is not. And
#  walking memory in order is fast while jumping through it is slow, so
#  reducing along the contiguous axis beats reducing across it; NumPy
#  handles the common cases, but it cannot repair a layout you chose badly.
#
#  The function to know is `np.ascontiguousarray(a)`: returns `a` itself if
#  it already qualifies, and a C-ordered copy otherwise. The trap is
#  `a.copy()`. Its default `order='K'` means "keep the input's layout", so
#  copying a transposed array gives you another column-major array, and it
#  always copies, even when nothing needed doing. `a.ravel()` has the same
#  keyword: `order='K'` walks memory in whatever order it is stored and so
#  never needs to copy; the default `order='C'` insists on row-major and
#  copies a Fortran array.
#
#  TASK
#    Fix both functions.
#
#  RUN IT
#    ./npt test 03_03
#
# =============================================================================

import numpy as np


def as_c_contiguous(a: np.ndarray) -> np.ndarray:
    """`a` laid out row-major in one block, copying only if it is not already."""
    # TODO: copy() keeps the input's memory order and always copies. There is
    # a function that does exactly what the docstring says.
    return a.copy()


def flat_in_memory_order(a: np.ndarray) -> np.ndarray:
    """The elements of `a` as a 1-D view in the order they sit in memory."""
    # TODO: order='C' insists on row-major and copies a Fortran array.
    # Ask for the order the array already has.
    return a.ravel(order="C")


def test_transposed_input_becomes_c_contiguous():
    m = np.arange(12.0).reshape(3, 4)
    t = m.T  # (4, 3), column-major
    assert not t.flags.c_contiguous
    c = as_c_contiguous(t)
    assert c.flags.c_contiguous
    np.testing.assert_array_equal(c, t)


def test_contiguous_input_is_not_copied():
    m = np.zeros((100, 100))
    assert as_c_contiguous(m) is m


def test_fortran_input_becomes_c_contiguous():
    f = np.asfortranarray(np.arange(6).reshape(2, 3))
    assert f.flags.f_contiguous and not f.flags.c_contiguous
    c = as_c_contiguous(f)
    assert c.flags.c_contiguous
    assert c.strides == (3 * c.itemsize, c.itemsize)


def test_flat_in_memory_order_of_a_fortran_array():
    f = np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]]))
    flat = flat_in_memory_order(f)
    np.testing.assert_array_equal(flat, [1, 4, 2, 5, 3, 6])
    assert np.shares_memory(flat, f)


def test_flat_in_memory_order_of_a_c_array():
    c = np.array([[1, 2, 3], [4, 5, 6]])
    flat = flat_in_memory_order(c)
    np.testing.assert_array_equal(flat, [1, 2, 3, 4, 5, 6])
    assert np.shares_memory(flat, c)
