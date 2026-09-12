# =============================================================================
#  01.05 -- Boolean masks
# =============================================================================
#
#  A comparison on an array is an array: `xs > 0` is a bool array the same
#  shape as `xs`. Index with it and you get the elements where it is True,
#  as a one-dimensional COPY. Assign through it and you write into the
#  original in place. Johansson, ch. 2, "Boolean Arrays and Conditional
#  Expressions" and "Fancy Indexing and Boolean-Valued Indexing".
#
#  The trap is Python's `and`, `or` and `not`. They are not operators you
#  can overload; they ask their operand "are you true?", and an array of
#  more than one element answers with
#
#      ValueError: The truth value of an array with more than one element is
#      ambiguous. Use a.any() or a.all()
#
#  For elementwise logic use `&`, `|` and `~`. They ARE overloaded, they are
#  bitwise on bool arrays (which is what you want), and they bind more
#  tightly than comparisons -- so `lo <= xs & xs < hi` parses as
#  `lo <= (xs & xs) < hi`. Put parentheses around each comparison.
#
#  A few idioms:
#
#      np.count_nonzero(mask)          how many True (a Python int)
#      mask.any(), mask.all()          collapse to one bool, deliberately
#      xs[mask] = value                write in place through the mask
#      np.where(mask, a, b)            choose elementwise (02.05)
#
#  TASK
#    Fix `in_range` and `count_outliers`, and write `clamp_negatives` so that
#    it returns a copy and leaves its input alone.
#
#  RUN IT
#    ./npt test 01_05
#
# =============================================================================

import numpy as np


def in_range(xs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The elements of `xs` with lo <= x < hi, in order."""
    # TODO: `and` asks the whole array for one truth value. Use `&` and
    # parenthesise each comparison.
    return xs[lo <= xs and xs < hi]


def count_outliers(xs: np.ndarray, k: float) -> int:
    """How many elements lie more than `k` standard deviations from the mean."""
    # TODO: `>` binds less tightly than `*`, but `|`/`&` bind MORE tightly than
    # `>`; this expression does not mean what it reads as. Also return a
    # Python int, not a NumPy bool or a 0-d array.
    mask = xs - xs.mean() > k * xs.std() | xs - xs.mean() < -k * xs.std()
    return mask.sum()


def clamp_negatives(xs: np.ndarray) -> np.ndarray:
    """A copy of `xs` with negative entries set to zero."""
    # TODO: copy, then assign through a mask.
    raise NotImplementedError


def test_in_range_uses_elementwise_logic():
    xs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_array_equal(in_range(xs, 1.0, 2.0), [1.0, 1.5])
    assert in_range(xs, 3.0, 4.0).shape == (0,)


def test_in_range_on_a_2d_array_flattens():
    # A boolean mask over a matrix selects elements, not rows: the result is
    # one-dimensional, in C order.
    m = np.array([[1, 5], [7, 2]])
    np.testing.assert_array_equal(in_range(m, 2, 7), [5, 2])


def test_count_outliers():
    rng = np.random.default_rng(0)
    xs = rng.normal(size=10_000)
    # For a standard normal about 0.27% of samples are beyond 3 sigma.
    n = count_outliers(xs, 3.0)
    assert isinstance(n, int)
    assert 10 <= n <= 50


def test_clamp_negatives_returns_a_copy():
    xs = np.array([-1.0, 2.0, -3.0])
    out = clamp_negatives(xs)
    np.testing.assert_array_equal(out, [0.0, 2.0, 0.0])
    np.testing.assert_array_equal(xs, [-1.0, 2.0, -3.0])


def test_masks_are_bool_arrays_you_can_combine():
    xs = np.arange(6)
    even = xs % 2 == 0
    big = xs > 2
    assert even.dtype == np.bool_
    np.testing.assert_array_equal(xs[even & big], [4])
    np.testing.assert_array_equal(xs[even | big], [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(xs[~even], [1, 3, 5])
