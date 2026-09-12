# =============================================================================
#  03.04 -- In-place operations
# =============================================================================
#
#  These two lines look alike and do different things:
#
#      total = total + delta      build a NEW array; point the name at it
#      total += delta             write into the array the name points at
#
#  In plain Python the difference is invisible for numbers, because numbers
#  are immutable and both lines rebind. For arrays it is the whole story.
#  The first line allocates, and inside a function it leaves the caller's
#  array untouched: `total` is a local name, rebinding it changes nothing
#  outside. The second line calls `np.add(total, delta, out=total)` (02.04)
#  and the caller sees the result, because there is only one array.
#
#  The same applies to views (03.01). `row = matrix[0]; row += 1` bumps the
#  first row of `matrix`, because `row` IS that memory. `row = row + 1`
#  makes a fresh array called `row` and `matrix` never hears about it.
#
#  In-place operations are how you avoid allocating (03.05) and how you
#  write functions that update their arguments, which is a legitimate
#  design when the docstring says so -- Effective Python, 3rd ed., Item 30:
#  "Know That Function Arguments Can Be Mutated". Two cautions. Take any
#  statistics you need BEFORE the first write, because after `xs -= mean`
#  the mean is zero. And an in-place op cannot change the dtype: an int
#  array cannot hold 0.5, so `ints += 0.5` raises rather than truncating,
#  which is the same rule as `out=` in 02.04.
#
#  TASK
#    Make all three functions modify their argument.
#
#  RUN IT
#    ./npt test 03_04
#
# =============================================================================

import numpy as np
import pytest


def accumulate(total: np.ndarray, delta: np.ndarray) -> None:
    """Add delta into total, in place."""
    # TODO: this rebinds the local name. The caller's array is untouched.
    total = total + delta


def bump_first_row(matrix: np.ndarray) -> None:
    """Add 1 to every element of the first row, in place."""
    # TODO: `row` starts as a view of the matrix, and then this line replaces
    # it with a new array. Add in place instead.
    row = matrix[0]
    row = row + 1


def standardise(xs: np.ndarray) -> None:
    """Shift and scale xs to mean 0 and std 1, in place."""
    # TODO: the statistics are right, but the result goes into a new array
    # and xs is never written.
    xs = (xs - xs.mean()) / xs.std()


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
