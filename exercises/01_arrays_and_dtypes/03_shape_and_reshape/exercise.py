# =============================================================================
#  01.03 -- Shape and reshape
# =============================================================================
#
#  `shape` is a tuple with one entry per axis, `ndim` is its length, `size`
#  is the product of its entries. A scalar array has shape `()`, ndim 0 and
#  size 1; a length-n vector has shape `(n,)`; the trailing comma matters.
#
#  `reshape` changes the shape without touching the data. That is possible
#  because of how the data is laid out: in C order (the default) the LAST
#  index varies fastest, so the 12 elements of an `arange(12)` read as rows
#  of a (3, 4) or as rows of a (4, 3) without moving a byte. Johansson,
#  ch. 2, "Order of Array Data in Memory" and "Reshaping and Resizing".
#
#  Two consequences:
#
#   * `reshape(-1, k)` fills in the -1 for you. Whether you write
#     `reshape(-1, k)` or `reshape(k, -1)` decides which elements end up
#     together, and both are valid shapes, so nothing will complain.
#
#   * A reshape that can be done by relabelling the same memory IS done that
#     way: the result is a view. `np.ravel(m)` and `m.reshape(-1)` return
#     views of `m` when they can; `m.flatten()` always copies. Modify the
#     view, and you have modified the original. Chapter 03 goes deeper; for
#     now the rule is: if a function promises not to change its input, make
#     the copy explicit.
#
#  TASK
#    Fix `as_batches` (rows of `batch` consecutive elements) and `centred_flat`
#    (must not modify `m`). `describe` is already correct.
#
#  RUN IT
#    ./npt test 01_03
#
# =============================================================================

import numpy as np


def describe(a: np.ndarray) -> tuple[int, tuple[int, ...], int]:
    """(number of axes, shape, total element count)."""
    return a.ndim, a.shape, a.size


def as_batches(xs: np.ndarray, batch: int) -> np.ndarray:
    """Split a flat sequence into rows of `batch` consecutive elements."""
    # TODO: this makes `batch` ROWS, not rows of `batch`. Consecutive
    # elements go across a row in C order, so batch is the last axis.
    return xs.reshape(batch, -1)


def centred_flat(m: np.ndarray) -> np.ndarray:
    """A flat copy of `m` with its mean subtracted. `m` is not modified."""
    # TODO: reshape(-1) is a view of `m` here, so `-=` writes through to the
    # caller's matrix. Make a copy that is a copy.
    flat = m.reshape(-1)
    flat -= flat.mean()
    return flat


def test_describe():
    assert describe(np.zeros((2, 3, 4))) == (3, (2, 3, 4), 24)
    assert describe(np.array(5.0)) == (0, (), 1)


def test_as_batches_keeps_consecutive_elements_together():
    xs = np.arange(12)
    batches = as_batches(xs, 4)
    assert batches.shape == (3, 4)
    np.testing.assert_array_equal(batches[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(batches[-1], [8, 9, 10, 11])


def test_reshape_is_a_view_of_the_same_data():
    # This is what makes reshape free, and what makes the next test matter.
    xs = np.arange(6)
    assert np.shares_memory(xs.reshape(2, 3), xs)


def test_centred_flat_does_not_modify_its_input():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    flat = centred_flat(m)
    np.testing.assert_allclose(flat, [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_array_equal(m, [[1.0, 2.0], [3.0, 4.0]])
    assert not np.shares_memory(flat, m)
