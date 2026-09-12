# =============================================================================
#  01.09 -- Scalars and Python numbers
# =============================================================================
#
#  Take one element out of an array and you do not get a Python number. You
#  get a NumPy scalar -- `np.float64`, `np.int64` -- which prints like a
#  number, mostly behaves like one, and differs in the two ways that matter:
#
#   * It has a fixed width. `np.int64(2**62) * 4` wraps to 0 (with a
#     RuntimeWarning for scalars; silently for arrays). A Python int would
#     have given 2**64. If you are computing something like a factorial, or
#     a hash, or anything that must be exact, keep it in Python ints:
#     `math.prod`, `sum` over a `range`, or `dtype=object` as a last resort.
#
#   * It carries its dtype into arithmetic. Under NumPy 2's promotion rules
#     (NEP 50; see the NumPy 2.0 migration guide) a PYTHON scalar is "weak":
#     `float32_array * 0.5` stays float32. A NUMPY scalar is strong:
#     `float32_array * np.float64(0.5)` becomes float64, and so does
#     everything downstream. This changed in 2.0 -- Johansson ch. 2 and older
#     tutorials describe NumPy 1.x, where the VALUE of a scalar decided --
#     and it is the reason to write `0.5`, not `np.float64(0.5)`, and to be
#     careful with values pulled out of float64 arrays and pushed into
#     float32 ones.
#
#  When you actually want a Python number -- to put in a JSON document, to
#  compare with `is`, to feed to code that checks `type(x) is float` -- ask
#  for it: `x.item()` converts a NumPy scalar or 1-element array to the
#  matching built-in type. `float(x)` and `int(x)` work too, but `.item()`
#  does not make you name the type.
#
#  TASK
#    Make `factorial` exact for n = 25, make `first_element` return a Python
#    float, and make `half` keep the dtype of its input.
#
#  RUN IT
#    ./npt test 01_09
#
# =============================================================================

import math  # noqa: F401  (you will want this)

import numpy as np


def factorial(n: int) -> int:
    """n! as an exact Python integer."""
    # TODO: np.prod over an int64 array wraps at 21!.
    return int(np.prod(np.arange(1, n + 1)))


def first_element(xs: np.ndarray) -> float:
    """The first element of `xs` as a plain Python float."""
    # TODO: this is a np.float64, not a float.
    return xs[0]


def half(xs: np.ndarray) -> np.ndarray:
    """`xs` scaled by one half, keeping its dtype."""
    # TODO: np.float64(0.5) is a strong scalar and promotes float32 to
    # float64. A Python literal is weak and keeps the array's dtype.
    return xs * np.float64(0.5)


def test_factorial_is_exact_beyond_int64():
    assert factorial(5) == 120
    assert factorial(20) == 2_432_902_008_176_640_000
    # 21! is larger than 2**63; an int64 product wraps to a negative number.
    assert factorial(25) == 15_511_210_043_330_985_984_000_000
    assert isinstance(factorial(3), int)


def test_numpy_integers_are_fixed_width():
    # Arrays do integer arithmetic in C: this wraps rather than growing.
    big = np.array([2**62], dtype=np.int64)
    assert (big * 4)[0] == 0
    # And a Python int that does not fit the dtype at all is refused.
    try:
        np.array([1], dtype=np.int64) + 2**70
    except OverflowError:
        pass
    else:
        raise AssertionError("expected OverflowError")


def test_first_element_is_a_python_float():
    xs = np.array([1.5, 2.5])
    x = first_element(xs)
    assert type(x) is float
    assert x == 1.5
    # An indexed element on its own is a NumPy scalar, not a Python one.
    assert type(xs[0]) is np.float64


def test_half_keeps_float32():
    xs = np.ones(4, dtype=np.float32)
    assert half(xs).dtype == np.float32


def test_nep_50_promotion_rules():
    # NumPy 2: a Python scalar is weak and adopts the array's dtype ...
    assert (np.float32(3) + 3.0).dtype == np.float32
    assert (np.array([1], dtype=np.int8) + 1).dtype == np.int8
    # ... but a NumPy scalar carries its own precision, and wins.
    assert (np.array([3], dtype=np.float32) + np.float64(3)).dtype == np.float64
