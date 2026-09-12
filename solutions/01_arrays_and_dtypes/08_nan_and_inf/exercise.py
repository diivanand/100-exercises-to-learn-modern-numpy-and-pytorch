# Solution -- 01.08 nan and inf
import numpy as np


def has_missing(xs: np.ndarray) -> bool:
    """True if any entry of `xs` is nan."""
    # nan is the one value that is not equal to itself, so `xs == np.nan` is
    # False everywhere. np.isnan is the only reliable question.
    return bool(np.isnan(xs).any())


def mean_ignoring_missing(xs: np.ndarray) -> float:
    """The mean of the non-nan entries of `xs`."""
    return float(np.nanmean(xs))


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b elementwise, with 0.0 wherever b is zero and no warnings."""
    # `where=` skips the division where b == 0; `out=` supplies the value
    # those positions keep. No division by zero happens, so nothing warns.
    out = np.zeros_like(a, dtype=np.float64)
    return np.divide(a, b, out=out, where=b != 0)


def test_has_missing():
    assert has_missing(np.array([1.0, np.nan, 3.0])) is True
    assert has_missing(np.array([1.0, 2.0, 3.0])) is False
    # inf is not nan; it is a value, just an unbounded one.
    assert has_missing(np.array([1.0, np.inf])) is False


def test_nan_is_not_equal_to_itself():
    x = np.nan
    assert x != x
    assert (np.array([np.nan]) == np.nan).any() is np.False_


def test_mean_ignoring_missing():
    assert mean_ignoring_missing(np.array([1.0, np.nan, 3.0])) == 2.0
    # An ordinary mean is poisoned by a single nan.
    assert np.isnan(np.array([1.0, np.nan, 3.0]).mean())


def test_safe_ratio_gives_zero_where_the_divisor_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 0.0, 4.0])
    np.testing.assert_array_equal(safe_ratio(a, b), [0.5, 0.0, 0.75])


def test_safe_ratio_does_not_warn():
    a = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    # Turn the "divide by zero" warning into an error for the duration of the
    # call: a division that happens here is a bug, not something to hide.
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(safe_ratio(a, b), [0.0, 0.0])
