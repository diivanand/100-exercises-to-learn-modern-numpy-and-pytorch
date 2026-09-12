# Solution -- 03.04 In-place operations
import numpy as np
import pytest


def accumulate(total: np.ndarray, delta: np.ndarray) -> None:
    """Add delta into total, in place."""
    # `total += delta` writes into the buffer `total` names. `total = total +
    # delta` builds a new array and points the LOCAL name at it; the caller's
    # array never changes and the new one is thrown away on return.
    total += delta


def bump_first_row(matrix: np.ndarray) -> None:
    """Add 1 to every element of the first row, in place."""
    # A row is a view (03.01), so an in-place op on the row lands in the
    # matrix. `row = row + 1` would rebind `row` to a fresh array instead.
    row = matrix[0]
    row += 1


def standardise(xs: np.ndarray) -> None:
    """Shift and scale xs to mean 0 and std 1, in place."""
    # Compute the statistics BEFORE the first write: after `xs -= mean`, the
    # std is unchanged (fine) but the mean is not, so order matters if you
    # ever reorganise this.
    mean = xs.mean()
    std = xs.std()
    xs -= mean
    xs /= std


def test_accumulate_updates_the_callers_array():
    total = np.zeros(3)
    accumulate(total, np.array([1.0, 2.0, 3.0]))
    accumulate(total, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(total, [2.0, 4.0, 6.0])


def test_bump_first_row():
    m = np.zeros((2, 3))
    bump_first_row(m)
    np.testing.assert_array_equal(m, [[1, 1, 1], [0, 0, 0]])


def test_standardise_in_place():
    rng = np.random.default_rng(34)
    xs = rng.normal(loc=5.0, scale=3.0, size=1000)
    before = xs
    standardise(xs)
    assert xs is before
    assert abs(xs.mean()) < 1e-12
    assert abs(xs.std() - 1.0) < 1e-12


def test_in_place_cannot_change_the_dtype():
    # Not a task. An integer array cannot hold 0.5, and += refuses rather
    # than silently truncating (the same rule as out= in 02.04).
    xs = np.arange(4)
    with pytest.raises(TypeError):
        xs += 0.5
    ys = xs + 0.5  # a new float array is fine
    assert ys.dtype.kind == "f"
