# Solution -- 02.05 where and select
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """max(x, 0), elementwise."""
    # `if x > 0` asks an array for a single truth value, which it refuses to
    # give. np.where evaluates the condition per element. (np.maximum(x, 0.0)
    # is the same thing and slightly faster; both are fine.)
    return np.where(x > 0, x, 0.0)


def grade(scores: np.ndarray) -> np.ndarray:
    """'A' for >= 90, 'B' for >= 75, 'C' for >= 50, otherwise 'F'."""
    # np.select takes the FIRST condition that matches, so the conditions
    # must go from most to least specific. The other way round, `>= 50`
    # would claim every A and B.
    conditions = [scores >= 90, scores >= 75, scores >= 50]
    return np.select(conditions, ["A", "B", "C"], default="F")


def safe_log(x: np.ndarray) -> np.ndarray:
    """log(x) where x > 0, and 0.0 elsewhere, without warnings."""
    # Both branches of np.where are computed for EVERY element, so
    # `np.where(x > 0, np.log(x), 0.0)` still evaluates log(0) and warns.
    # Feed log a safe input instead, then choose.
    positive = x > 0
    return np.where(positive, np.log(np.where(positive, x, 1.0)), 0.0)


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
