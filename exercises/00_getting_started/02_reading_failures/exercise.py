# =============================================================================
#  00.02 -- Reading failures
# =============================================================================
#
#  The loop of this course is `./npt next`: it runs the first exercise that
#  is not yet passing and shows you the failure. Most of the skill is in
#  reading that output, so this exercise is about the output.
#
#  A failing `assert` in pytest looks like this:
#
#      >       assert mean_of_positives(np.array([0.0, 2.0])) == 2.0
#      E       assert 1.0 == 2.0
#      E        +  where 1.0 = mean_of_positives(array([0., 2.]))
#
#  The `>` line is the assertion as written; the `E` lines are pytest
#  rewriting it with the values substituted in, innermost call first. You
#  rarely need a debugger: the wrong value and the input that produced it
#  are already on the screen.
#
#  Arrays are compared with the helpers in `np.testing`, never with `==` (an
#  array `==` is itself an array, and `assert` on an array is an error --
#  01.05 explains why). `np.testing.assert_array_equal(a, b)` and
#  `np.testing.assert_allclose(a, b, rtol=...)` print the shapes, the number
#  of mismatched elements, and the largest absolute and relative differences.
#  Read those numbers: "1 mismatched element, max abs diff 1e-7" is a rounding
#  question; "all 1000 mismatched, max diff 1e3" is a wrong formula.
#
#  Two more habits worth forming now:
#
#   * Decide what the empty case means, and write it down. A mean of nothing
#     is nan in NumPy (plus a RuntimeWarning), and nan silently poisons every
#     later sum it takes part in. Effective Python, 3rd ed., Item 81 ("assert
#     Internal Assumptions and raise Missed Expectations") is about making
#     such decisions explicit rather than letting them happen.
#   * Do not modify what you were given unless the name says so. NumPy makes
#     that easy to get wrong, because so many operations return views of the
#     original data (chapter 03).
#
#  TASK
#    Fix `mean_of_positives` so that zero does not count as positive and an
#    input with no positive entries gives 0.0 rather than nan. Then run the
#    tests and read the output for `positive_part`, which is correct: notice
#    how a passing test says nothing at all.
#
#  RUN IT
#    ./npt test 00_02
#
# =============================================================================

import numpy as np


def mean_of_positives(xs: np.ndarray) -> float:
    """The mean of the strictly positive entries of `xs`, or 0.0 if there are none."""
    # TODO: `>=` lets zero in, and the mean of an empty selection is nan.
    positives = xs[xs >= 0]
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
