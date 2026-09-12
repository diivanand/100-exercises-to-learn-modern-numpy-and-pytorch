# Solution -- 01.05 Boolean masks
import numpy as np


def in_range(xs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The elements of `xs` with lo <= x < hi, in order."""
    # `&`, not `and`: `and` asks each array for a single truth value, which
    # an array with more than one element refuses to give. The parentheses
    # are required because `&` binds tighter than `<=`.
    return xs[(lo <= xs) & (xs < hi)]


def count_outliers(xs: np.ndarray, k: float) -> int:
    """How many elements lie more than `k` standard deviations from the mean."""
    mask = np.abs(xs - xs.mean()) > k * xs.std()
    # count_nonzero is the idiomatic count of True values; mask.sum() also
    # works but produces an integer array scalar rather than a Python int.
    return int(np.count_nonzero(mask))


def clamp_negatives(xs: np.ndarray) -> np.ndarray:
    """A copy of `xs` with negative entries set to zero."""
    out = xs.copy()
    out[out < 0] = 0  # boolean-mask assignment writes into `out` in place
    return out


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
