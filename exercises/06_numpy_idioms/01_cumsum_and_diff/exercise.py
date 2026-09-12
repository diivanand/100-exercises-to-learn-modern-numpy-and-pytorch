# =============================================================================
#  06.01 -- cumsum and diff
# =============================================================================
#
#  `np.cumsum` turns a sequence into its running total, and `np.diff` turns
#  a running total back into a sequence. Together they are the array
#  version of integration and differentiation, and the basis of a family of
#  tricks that replace a loop over windows with two slices.
#
#  The one to know: the sum of x[i:i+w] is c[i+w] - c[i], where c is the
#  cumulative sum with a ZERO PREPENDED, so that c[i] is "the sum of the
#  first i elements". Every window sum is then one vectorised subtraction,
#  and a moving average is that divided by w. Leave the zero out and the
#  arithmetic still runs, silently missing the first window and returning
#  one result too few. Johansson, *Numerical Python* ch. 2, tabulates
#  np.cumsum with the other aggregates; ch. 19 uses it as the benchmark
#  workload.
#
#  `np.diff(x, prepend=v)` puts v in front before differencing, so the
#  output has the same length as x and the first entry is x[0] - v. That
#  is what makes diff exactly invertible by cumsum.
#
#  TASK
#    Fix the off-by-one in `moving_average` (read the second test), and
#    implement `changes` and `reconstruct` as inverses.
#
#  RUN IT
#    ./npt test 06_01
#
# =============================================================================

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of each run of `window` consecutive values: len(x) - window + 1 results."""
    # TODO: c[i] here is the sum of the first i+1 elements, not the first i,
    # so c[window:] - c[:-window] skips the first window and is one short.
    c = np.cumsum(x, dtype=np.float64)
    return (c[window:] - c[:-window]) / window


def changes(x: np.ndarray) -> np.ndarray:
    """x[i] - x[i-1] for every i, with the first change taken from 0."""
    # TODO: np.diff with prepend=0.0 gives the same length as x.
    return np.diff(x)


def reconstruct(changes_: np.ndarray) -> np.ndarray:
    """Undo `changes`: the series whose changes-from-zero are `changes_`."""
    # TODO: the inverse of diff-from-zero is cumsum.
    return changes_


def test_moving_average_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    for w in (1, 3, 7):
        expected = np.array([x[i : i + w].mean() for i in range(len(x) - w + 1)])
        np.testing.assert_allclose(moving_average(x, w), expected)


def test_moving_average_includes_the_first_window():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(moving_average(x, 2), [1.5, 2.5, 3.5, 4.5])
    assert moving_average(x, 5).shape == (1,)
    np.testing.assert_allclose(moving_average(x, 5), [3.0])


def test_changes_and_reconstruct_are_inverses():
    rng = np.random.default_rng(1)
    x = rng.integers(-5, 5, size=50).astype(float)
    d = changes(x)
    assert d.shape == x.shape
    assert d[0] == x[0]
    np.testing.assert_allclose(reconstruct(d), x)
