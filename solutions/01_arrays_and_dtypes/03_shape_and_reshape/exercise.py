# Solution -- 01.03 Shape and reshape
import numpy as np


def describe(a: np.ndarray) -> tuple[int, tuple[int, ...], int]:
    """(number of axes, shape, total element count)."""
    return a.ndim, a.shape, a.size


def as_batches(xs: np.ndarray, batch: int) -> np.ndarray:
    """Split a flat sequence into rows of `batch` consecutive elements."""
    # -1 means "work this one out". Rows of `batch`, so batch is the LAST
    # axis: consecutive elements go across a row in C order.
    return xs.reshape(-1, batch)


def centred_flat(m: np.ndarray) -> np.ndarray:
    """A flat copy of `m` with its mean subtracted. `m` is not modified."""
    # np.ravel / reshape(-1) return a view whenever they can, so subtracting
    # in place would write through to the caller's matrix. flatten() always
    # copies, which is exactly the guarantee this function makes.
    flat = m.flatten()
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
