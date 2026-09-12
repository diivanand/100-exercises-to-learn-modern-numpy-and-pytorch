# Solution -- 06.07 NumPy 2 migration

import numpy as np

# np.float_ was an alias for np.float64 and is gone; say what you mean.
DTYPE = np.float64


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal-rule integral of y over x."""
    # np.trapz -> np.trapezoid (renamed in 2.0 to match the rest of the
    # scientific-Python ecosystem).
    return float(np.trapezoid(y, x))


def is_member(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    # np.in1d -> np.isin (05.02).
    return np.isin(values, allowed)


def product_of(x: np.ndarray) -> float:
    # np.product -> np.prod (likewise cumproduct -> cumprod, alltrue -> all).
    return float(np.prod(x))


def as_array(x: object) -> np.ndarray:
    """`x` as a float64 array, without copying when `x` already is one."""
    # np.array(x, copy=False) now RAISES if a copy is unavoidable (a list
    # must be copied into an array). np.asarray means "copy only if needed".
    return np.asarray(x, dtype=DTYPE)


def spread(x: np.ndarray) -> float:
    # The ndarray.ptp method was removed; the function np.ptp remains.
    return float(np.ptp(x))


def noisy(n: int, seed: int) -> np.ndarray:
    # np.random.seed / np.random.normal -> a Generator (04.05).
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).astype(DTYPE)


def test_dtype_is_float64():
    assert DTYPE is np.float64


def test_integrate():
    x = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(integrate(x**2, x), 1.0 / 3.0, atol=1e-4)


def test_is_member():
    np.testing.assert_array_equal(
        is_member(np.array([1, 5, 3]), np.array([3, 4, 5])), [False, True, True]
    )


def test_product_of():
    assert product_of(np.array([2.0, 3.0, 4.0])) == 24.0


def test_as_array_copies_a_list_and_not_an_array():
    from_list = as_array([1, 2, 3])
    assert from_list.dtype == np.float64
    np.testing.assert_array_equal(from_list, [1.0, 2.0, 3.0])
    existing = np.array([4.0, 5.0])
    assert as_array(existing) is existing


def test_spread():
    assert spread(np.array([3.0, -1.0, 7.0])) == 8.0


def test_noisy_is_reproducible_without_global_state():
    a = noisy(10, seed=3)
    b = noisy(10, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float64
    before = np.random.get_bit_generator().state["state"]["key"].copy()
    noisy(10, seed=4)
    np.testing.assert_array_equal(
        np.random.get_bit_generator().state["state"]["key"], before
    )
