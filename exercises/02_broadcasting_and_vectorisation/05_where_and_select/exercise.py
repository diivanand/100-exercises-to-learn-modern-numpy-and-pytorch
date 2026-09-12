# =============================================================================
#  02.05 -- where and select
# =============================================================================
#
#  `if x > 0:` does not work on an array. `x > 0` is an array of booleans,
#  and Python's `if` wants one truth value, so NumPy raises: "The truth
#  value of an array with more than one element is ambiguous." The
#  elementwise conditional is
#
#      np.where(condition, if_true, if_false)
#
#  which picks, per element, from the two alternatives (both broadcast to
#  the condition's shape). For more than two cases there is
#
#      np.select([cond_1, cond_2, ...], [value_1, value_2, ...], default=...)
#
#  which takes the FIRST condition that holds. Johansson, Numerical Python
#  3rd ed., ch. 2 "Boolean Arrays and Conditional Expressions" shows both.
#
#  Two things bite. Because np.select takes the first match, the order of
#  the conditions is part of the logic: `scores >= 50` listed before
#  `scores >= 90` claims every A. And np.where is a function call, so BOTH
#  alternatives are fully evaluated before it chooses. `np.where(x > 0,
#  np.log(x), 0.0)` still computes log(0) and log(-3): the result is right,
#  but a RuntimeWarning is printed every time, and in code that runs under
#  `np.errstate(all="raise")` it is an exception. Make the input safe before
#  the risky call, then select.
#
#  (`np.where(condition)` with one argument returns indices; that form is
#  better spelled `np.nonzero(condition)`, and is not what this is about.)
#
#  TASK
#    Fix the three functions.
#
#  RUN IT
#    ./npt test 02_05
#
# =============================================================================

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """max(x, 0), elementwise."""
    # TODO: an `if` on an array does not compile down to anything; it raises.
    if x > 0:
        return x
    return 0.0


def grade(scores: np.ndarray) -> np.ndarray:
    """'A' for >= 90, 'B' for >= 75, 'C' for >= 50, otherwise 'F'."""
    # TODO: np.select takes the first condition that holds. Order these from
    # the most specific to the least.
    conditions = [scores >= 50, scores >= 75, scores >= 90]
    return np.select(conditions, ["C", "B", "A"], default="F")


def safe_log(x: np.ndarray) -> np.ndarray:
    """log(x) where x > 0, and 0.0 elsewhere, without warnings."""
    # TODO: np.log(x) is evaluated for EVERY element before np.where picks.
    # Give log an input that is safe everywhere, then choose.
    return np.where(x > 0, np.log(x), 0.0)


def test_relu():
    np.testing.assert_allclose(relu(np.array([-2.0, 0.0, 3.5])), [0.0, 0.0, 3.5])
    assert relu(np.array([[-1.0, 1.0]])).shape == (1, 2)


def test_grade():
    scores = np.array([95, 90, 89, 75, 74, 50, 49, 0])
    np.testing.assert_array_equal(grade(scores), ["A", "A", "B", "B", "C", "C", "F", "F"])


def test_grade_boundaries_are_inclusive():
    assert grade(np.array([90]))[0] == "A"
    assert grade(np.array([75]))[0] == "B"
    assert grade(np.array([50]))[0] == "C"


def test_safe_log_values():
    out = safe_log(np.array([np.e, 1.0, 0.0, -3.0]))
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 0.0])


def test_safe_log_is_silent():
    # np.errstate(all="raise") turns a would-be RuntimeWarning into an
    # exception, so a stray log(0) fails the test instead of scrolling past.
    with np.errstate(all="raise"):
        safe_log(np.array([0.0, -1.0, 2.0]))
