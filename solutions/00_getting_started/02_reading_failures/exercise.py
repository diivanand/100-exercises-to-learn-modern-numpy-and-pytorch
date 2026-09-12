# Solution -- 00.02 Reading failures
import numpy as np


def mean_of_positives(xs: np.ndarray) -> float:
    """The mean of the strictly positive entries of `xs`, or 0.0 if there are none."""
    positives = xs[xs > 0]
    # An empty mean is nan (and a RuntimeWarning), which is rarely what a
    # caller wants to carry on computing with. Decide the empty case explicitly.
    if positives.size == 0:
        return 0.0
    return float(positives.mean())


def positive_part(xs: np.ndarray) -> np.ndarray:
    """`xs` with every negative entry replaced by zero. Never modifies `xs`."""
    return np.maximum(xs, 0)


def test_mean_of_positives():
    assert mean_of_positives(np.array([1.0, -2.0, 3.0])) == 2.0
    assert mean_of_positives(np.array([4.0, 4.0])) == 4.0


def test_zero_is_not_positive():
    # A plausible shortcut is `xs >= 0`. Zero is not positive, and including it
    # changes the answer.
    assert mean_of_positives(np.array([0.0, 2.0])) == 2.0


def test_no_positives_gives_zero_not_nan():
    result = mean_of_positives(np.array([-1.0, -2.0]))
    assert result == 0.0
    assert mean_of_positives(np.array([])) == 0.0


def test_positive_part_is_elementwise():
    xs = np.array([-1.5, 0.0, 2.5, -0.1])
    np.testing.assert_array_equal(positive_part(xs), [0.0, 0.0, 2.5, 0.0])
    # The input is left alone.
    np.testing.assert_array_equal(xs, [-1.5, 0.0, 2.5, -0.1])
