# Solution -- 03.03 Contiguity and memory order
import numpy as np


def as_c_contiguous(a: np.ndarray) -> np.ndarray:
    """`a` laid out row-major in one block, copying only if it is not already."""
    # np.ascontiguousarray returns its input untouched when it already
    # qualifies and copies otherwise. `a.copy()` would always copy, and worse,
    # its default order 'K' KEEPS the input's layout, so copying a transposed
    # array gives you another column-major array.
    return np.ascontiguousarray(a)


def flat_in_memory_order(a: np.ndarray) -> np.ndarray:
    """The elements of `a` as a 1-D view in the order they sit in memory."""
    # order='K' means "however it is stored", so a Fortran-ordered array is
    # walked column by column and no copy is needed. The default order='C'
    # would insist on row-major and copy.
    return a.ravel(order="K")


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
