# =============================================================================
#  01.08 -- nan and inf
# =============================================================================
#
#  IEEE floating point has two kinds of non-number. `inf` is a value: it
#  compares larger than everything, and `1.0 / 0.0` produces it (NumPy emits
#  a RuntimeWarning and carries on, where Python would raise). `nan` -- "not
#  a number" -- is the result of a computation with no answer: `0.0 / 0.0`,
#  `inf - inf`, `sqrt(-1.0)`. NumPy also uses nan to mean "missing", because
#  a float array has no other way to say it.
#
#  nan has one defining property: it is not equal to anything, INCLUDING
#  ITSELF. So `xs == np.nan` is False everywhere, `nan in xs` is False, and
#  the only way to ask the question is `np.isnan(xs)`. Any arithmetic with
#  nan yields nan, so one missing reading turns a whole `mean()` into nan;
#  the `nan`-prefixed reductions (`np.nanmean`, `np.nansum`, `np.nanmax`)
#  skip them instead. Johansson, ch. 2, "Data Types" shows nan appearing
#  from `np.sqrt` of a negative; chapter 06.06 returns to floating point.
#
#  Division by zero is where most nans and infs are born. NumPy's ufuncs
#  take two arguments that let you avoid the division rather than clean up
#  after it:
#
#      np.divide(a, b, out=np.zeros_like(a), where=b != 0)
#
#  `where=` masks the operation; `out=` says what the masked positions hold.
#  And `np.errstate(divide="raise")` turns the warning into an exception for
#  the code inside the `with`, which is the right setting for a test.
#
#  TASK
#    Fix `has_missing` and `mean_ignoring_missing`, then write `safe_ratio`
#    so that it never divides by zero (the last test checks that with
#    errstate, not by inspecting the output).
#
#  RUN IT
#    ./npt test 01_08
#
# =============================================================================

import numpy as np


def has_missing(xs: np.ndarray) -> bool:
    """True if any entry of `xs` is nan."""
    # TODO: nan == nan is False. This function can never return True.
    return bool((xs == np.nan).any())


def mean_ignoring_missing(xs: np.ndarray) -> float:
    """The mean of the non-nan entries of `xs`."""
    # TODO: one nan makes the whole mean nan.
    return float(xs.mean())


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b elementwise, with 0.0 wherever b is zero and no warnings."""
    # TODO: use np.divide with `out=` and `where=`.
    raise NotImplementedError


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
