# Solution -- 01.09 Scalars and Python numbers
import math

import numpy as np


def factorial(n: int) -> int:
    """n! as an exact Python integer."""
    # NumPy integers are 64-bit and wrap; 21! already does not fit. Python
    # ints are arbitrary precision, so keep the arithmetic in Python.
    return math.prod(range(1, n + 1))


def first_element(xs: np.ndarray) -> float:
    """The first element of `xs` as a plain Python float."""
    # .item() converts a NumPy scalar (or 1-element array) to the matching
    # Python type. `float(xs[0])` also works; .item() also handles ints and
    # complex without you naming the type.
    return xs[0].item()


def half(xs: np.ndarray) -> np.ndarray:
    """`xs` scaled by one half, keeping its dtype."""
    # A Python float is "weak" under NEP 50: it takes the array's dtype. A
    # np.float64(0.5) would be a strong scalar and would promote float32
    # arrays to float64, doubling the memory of every result.
    return xs * 0.5


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
