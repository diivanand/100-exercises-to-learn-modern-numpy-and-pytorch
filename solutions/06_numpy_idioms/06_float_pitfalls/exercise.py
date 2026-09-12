# Solution -- 06.06 Floating-point pitfalls

import numpy as np


def round_half_up(x: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, with .5 going UP (2.5 -> 3, -2.5 -> -2)."""
    # np.round is round-half-to-EVEN (IEEE 754's default): 0.5 -> 0, 1.5 -> 2,
    # 2.5 -> 2. That is the statistically unbiased choice, and not what a
    # spreadsheet user means by "round". floor(x + 0.5) is half-up.
    return np.floor(x + 0.5)


def variance(x: np.ndarray) -> np.floating:
    """The population variance of `x`, computed in x's own dtype."""
    # Two passes: subtract the mean first, then square. The one-pass formula
    # E[x^2] - E[x]^2 subtracts two nearly equal large numbers and loses
    # every significant digit of the small difference we want.
    centred = x - x.mean()
    return (centred * centred).mean()


def equal_enough(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise 'a and b agree to floating-point precision'."""
    # np.isclose is the elementwise form of assert_allclose (06.05).
    return np.isclose(a, b, rtol=1e-7, atol=1e-12)


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
