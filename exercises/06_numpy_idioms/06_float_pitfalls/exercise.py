# =============================================================================
#  06.06 -- Floating-point pitfalls
# =============================================================================
#
#  Three things about IEEE 754 arithmetic that bite people who know it is
#  "approximate" and think that is the whole story.
#
#  1. ROUNDING IS HALF-TO-EVEN. `np.round(2.5)` is 2, and `np.round(3.5)`
#     is 4. Python's built-in round does the same. This is the IEEE default
#     because it is unbiased over many roundings; it is also not what your
#     accountant means. If you need half-up, say so: floor(x + 0.5).
#
#  2. CATASTROPHIC CANCELLATION. Subtracting two nearly equal numbers keeps
#     the difference and throws away the shared leading digits, which were
#     the accurate ones. The textbook example is the one-pass variance
#     E[x^2] - E[x]^2: for data with a large mean, both terms are huge and
#     the answer is their small difference, so it comes out as noise or
#     negative. The two-pass formula (subtract the mean, then square) does
#     not have the problem. Kirk & Hwu, *PMPP* §7.5, "Algorithm
#     Considerations", tells the same story for reductions on the GPU:
#     summation ORDER changes the result, and sorting or grouping values of
#     similar magnitude before adding keeps precision. NumPy's np.sum uses
#     pairwise summation for the same reason, which is why a Python loop
#     summing float32 drifts and np.sum does not.
#
#  3. EQUALITY IS THE WRONG QUESTION. Use `np.isclose` (elementwise) or
#     assert_allclose (06.05), with tolerances chosen on purpose.
#
#  TASK
#    Make the three functions honour their docstrings; the variance test
#    uses float32 with an offset of ten thousand and will not pass a
#    one-pass formula.
#
#  RUN IT
#    ./npt test 06_06
#
# =============================================================================

import numpy as np


def round_half_up(x: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, with .5 going UP (2.5 -> 3, -2.5 -> -2)."""
    # TODO: np.round is half-to-even.
    return np.round(x)


def variance(x: np.ndarray) -> np.floating:
    """The population variance of `x`, computed in x's own dtype."""
    # TODO: E[x^2] - E[x]^2 cancels catastrophically for data far from zero.
    # Subtract the mean first.
    return (x * x).mean() - x.mean() ** 2


def equal_enough(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise 'a and b agree to floating-point precision'."""
    # TODO: np.isclose with explicit rtol and atol.
    return a == b


def test_round_half_up():
    x = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 2.4, 2.6])
    np.testing.assert_array_equal(round_half_up(x), [1, 2, 3, 0, -2, 2, 3])


def test_variance_survives_a_large_offset_in_float32():
    # Unit-variance noise sitting on top of 10 000. In float32, x*x is about
    # 1e8 with a resolution of 8, so E[x^2] - E[x]^2 has no digits left; the
    # two-pass formula is exact to the precision of the noise.
    rng = np.random.default_rng(0)
    x = (10_000.0 + rng.standard_normal(100_000)).astype(np.float32)
    v = variance(x)
    assert np.dtype(v.dtype) == np.float32
    assert abs(float(v) - 1.0) < 0.02


def test_variance_matches_numpy_in_float64():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000) * 3.0 + 7.0
    np.testing.assert_allclose(variance(x), np.var(x))


def test_equal_enough():
    a = np.array([0.1 + 0.2, 1.0, 2.0])
    b = np.array([0.3, 1.0 + 1e-3, 2.0])
    np.testing.assert_array_equal(equal_enough(a, b), [True, False, True])
