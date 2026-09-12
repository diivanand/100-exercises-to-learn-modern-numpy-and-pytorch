# =============================================================================
#  06.07 -- NumPy 2 migration
# =============================================================================
#
#  NumPy 2.0 (June 2024) was the first major release in eighteen years, and
#  it removed about a hundred names from the main namespace, most of them
#  aliases and duplicates that had been deprecated for years. Code written
#  against the books on your shelf -- McKinney (2012), the earlier
#  Johansson editions, most blog posts -- trips over some of them. This
#  file is such a piece of code. Every function in it was fine on NumPy
#  1.26 and at least one line of it is wrong on 2.x. The migration guide
#  (numpy.org, "NumPy 2.0 migration guide") is the reference; the changes
#  you will meet most often:
#
#      np.float_, np.complex_        ->  np.float64, np.complex128
#      np.NaN, np.Inf, np.infty      ->  np.nan, np.inf
#      np.in1d                       ->  np.isin
#      np.trapz                      ->  np.trapezoid
#      np.product, np.cumproduct     ->  np.prod, np.cumprod
#      np.alltrue, np.sometrue       ->  np.all, np.any
#      np.row_stack                  ->  np.vstack
#      arr.ptp(), arr.newbyteorder() ->  np.ptp(arr), arr.view(dtype.newbyteorder())
#      np.array(x, copy=False)       ->  np.asarray(x)   (copy=False now RAISES
#                                        when a copy is unavoidable)
#      np.random.seed / rand / ...   ->  np.random.default_rng(seed)  (04.05;
#                                        still present, but legacy)
#
#  And one that is not a rename: NEP 50 changed type promotion so that
#  NumPy scalars keep their precision (`np.float32(3) + 3.0` is float32 now;
#  01.09 covers it). `ruff check --select NPY201` finds most of the renames
#  mechanically; this course's ruff configuration includes it.
#
#  TASK
#    Bring the file to NumPy 2.5. Each function has one thing to fix; the
#    tests describe the intended behaviour.
#
#  NOTE  This exercise starts as an error at import time: `np.float_` no
#        longer exists, so pytest cannot even collect the tests until the
#        first line is fixed. The others fail one at a time after that.
#
#  RUN IT
#    ./npt test 06_07
#
# =============================================================================

import numpy as np

# TODO: np.float_ was removed in NumPy 2.0.
DTYPE = np.float_  # noqa: NPY201


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal-rule integral of y over x."""
    # TODO: renamed.
    return float(np.trapz(y, x))  # noqa: NPY201


def is_member(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    # TODO: removed; see 05.02.
    return np.in1d(values, allowed)  # noqa: NPY201


def product_of(x: np.ndarray) -> float:
    # TODO: removed alias.
    return float(np.product(x))  # noqa: NPY003, NPY201


def as_array(x: object) -> np.ndarray:
    """`x` as a float64 array, without copying when `x` already is one."""
    # TODO: copy=False now means "never copy" and raises for a list.
    return np.array(x, dtype=DTYPE, copy=False)


def spread(x: np.ndarray) -> float:
    # TODO: the method is gone; the function is not.
    return float(x.ptp())


def noisy(n: int, seed: int) -> np.ndarray:
    # TODO: legacy global generator.
    np.random.seed(seed)
    return np.random.normal(size=n).astype(DTYPE)


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
