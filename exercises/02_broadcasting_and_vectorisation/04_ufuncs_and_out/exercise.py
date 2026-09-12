# =============================================================================
#  02.04 -- ufuncs and out=
# =============================================================================
#
#  `np.add`, `np.multiply`, `np.exp`, `np.maximum`, ... are "universal
#  functions": one compiled elementwise kernel each, with broadcasting,
#  type promotion and a handful of methods bolted on. The operators are just
#  spellings of them: `a + b` calls `np.add(a, b)`. Knowing that unlocks the
#  methods (McKinney, Python for Data Analysis, ch. 12 "Advanced ufunc
#  usage"; Johansson, Numerical Python 3rd ed., ch. 2 "Elementwise
#  Functions"):
#
#      np.add.reduce(xs)          the sum          (np.sum is this)
#      np.add.accumulate(xs)      prefix sums      (np.cumsum is this)
#      np.multiply.outer(a, b)    the (n, m) table a[i] * b[j]
#      np.maximum.reduce(m, axis=0)
#
#  and the keyword every ufunc accepts: `out=`. It names an existing array
#  to write the result into, instead of allocating a new one. Two things
#  follow. First, `out=` is how you update the CALLER's array from inside a
#  function: `np.add(total, delta, out=total)` writes into `total`, where
#  `total + delta` builds a new array the caller never sees (03.04 has more).
#  Second, the output buffer must be able to hold the result. Ufuncs cast
#  with the 'same_kind' rule: float to float, int to int, but never float
#  down to int. Hand np.divide an integer `out` and it raises rather than
#  truncating -- which is right, and is the bug in `normalised` below.
#
#  TASK
#    Fix `normalised` and `add_into`.
#
#  RUN IT
#    ./npt test 02_04
#
# =============================================================================

import numpy as np
import pytest


def running_total(xs: np.ndarray) -> np.ndarray:
    """Prefix sums: out[i] = xs[0] + ... + xs[i]."""
    return np.add.accumulate(xs)


def multiplication_table(n: int) -> np.ndarray:
    """The (n, n) table t[i, j] = (i + 1) * (j + 1)."""
    values = np.arange(1, n + 1)
    return np.multiply.outer(values, values)


def normalised(xs: np.ndarray) -> np.ndarray:
    """xs divided by its largest absolute value, as float64."""
    # TODO: np.empty_like copies xs's dtype. When xs holds integers the
    # buffer cannot receive a float result and np.divide raises. Say the
    # dtype the result needs.
    out = np.empty_like(xs)
    np.divide(xs, np.abs(xs).max(), out=out)
    return out


def add_into(total: np.ndarray, delta: np.ndarray) -> None:
    """total += delta, without allocating a temporary."""
    # TODO: this builds a new array and binds the LOCAL name `total` to it.
    # The caller's array is untouched. Write into it with out=.
    total = np.add(total, delta)


def test_running_total():
    np.testing.assert_array_equal(running_total(np.array([1, 2, 3, 4])), [1, 3, 6, 10])
    np.testing.assert_array_equal(running_total(np.array([2.5])), [2.5])


def test_multiplication_table():
    t = multiplication_table(4)
    np.testing.assert_array_equal(t[3], [4, 8, 12, 16])
    assert t.shape == (4, 4)
    assert t[2, 1] == 6


def test_normalised_from_integers():
    out = normalised(np.array([1, -4, 2]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.25, -1.0, 0.5])


def test_normalised_from_floats():
    np.testing.assert_allclose(normalised(np.array([0.5, 0.25])), [1.0, 0.5])


def test_add_into_updates_the_callers_array():
    total = np.zeros(3)
    delta = np.array([1.0, 2.0, 3.0])
    add_into(total, delta)
    add_into(total, delta)
    np.testing.assert_allclose(total, [2.0, 4.0, 6.0])


def test_out_must_be_the_right_dtype():
    # Not a task: a demonstration of the rule normalised() has to respect.
    buffer = np.empty(2, dtype=np.int64)
    with pytest.raises(TypeError):
        np.divide(np.array([1, 2]), 4, out=buffer)
