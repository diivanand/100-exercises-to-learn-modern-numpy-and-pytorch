# =============================================================================
#  03.05 -- Avoiding temporaries
# =============================================================================
#
#  `a * x + b` on arrays of a million doubles allocates a million doubles
#  for `a * x`, then a million more for the sum, then frees the first. The
#  arithmetic is trivial; the traffic is not. Each temporary is written to
#  memory and read back, and for arrays larger than the cache that traffic
#  IS the running time. NumPy has no fusion: it cannot see the whole
#  expression, only one operator at a time.
#
#  When it matters -- inside a loop, on big arrays -- you fuse by hand with
#  the `out=` keyword from 02.04, reusing one buffer:
#
#      np.multiply(x, a, out=out)
#      np.add(out, b, out=out)          no temporaries at all
#
#  and you pick the operation that does not need one. `np.sum(x**2)`
#  builds x**2; `np.dot(x, x)` streams through x once with a running
#  total, and for floats it is a BLAS call. Johansson, Numerical Python 3rd
#  ed., ch. 19 "Code Optimization" starts from exactly this kind of
#  expression before reaching for Numba.
#
#  How do the tests know? NumPy registers its array buffers with Python's
#  `tracemalloc` (Effective Python, 3rd ed., Item 115 introduces the module
#  for Python objects), so the peak traced memory across a call includes
#  every temporary NumPy allocated. The tests require the peak to be under
#  a tenth of one array. Do not reach for this blindly, though: 03.06 is
#  about measuring first, and most code is not on this path.
#
#  TASK
#    Make `affine_into` write into `out` without temporaries, and make
#    `squared_norm` not build x * x.
#
#  RUN IT
#    ./npt test 03_05
#
# =============================================================================

import tracemalloc

import numpy as np


def affine_into(x: np.ndarray, a: float, b: float, out: np.ndarray) -> np.ndarray:
    """out = a * x + b, written into `out` with no temporary array."""
    # TODO: two full-size temporaries, and `out` is rebound rather than
    # written. Use out= on both ufunc calls.
    out = a * x + b
    return out


def squared_norm(x: np.ndarray) -> float:
    """sum(x * x) without building x * x."""
    # TODO: x**2 is a full-size temporary. A dot product is not.
    return float(np.sum(x**2))


def _peak_bytes(fn) -> int:
    # NumPy registers its array buffers with tracemalloc, so the peak here
    # includes every temporary NumPy allocates on the way. (Effective Python,
    # 3rd ed., Item 115 covers tracemalloc for Python objects; NumPy opts in.)
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_affine_into_values():
    x = np.array([0.0, 1.0, 2.0])
    out = np.empty(3)
    result = affine_into(x, 2.0, 1.0, out)
    np.testing.assert_allclose(out, [1.0, 3.0, 5.0])
    assert result is out


def test_affine_into_allocates_nothing():
    x = np.ones(1_000_000)
    out = np.empty_like(x)
    peak = _peak_bytes(lambda: affine_into(x, 3.0, -1.0, out))
    assert peak < x.nbytes // 10, f"{peak} bytes of temporaries"
    np.testing.assert_allclose(out, 2.0)


def test_squared_norm_value():
    assert squared_norm(np.array([3.0, 4.0])) == 25.0
    assert squared_norm(np.zeros(5)) == 0.0


def test_squared_norm_allocates_nothing():
    x = np.full(1_000_000, 0.5)
    peak = _peak_bytes(lambda: squared_norm(x))
    assert peak < x.nbytes // 10, f"{peak} bytes of temporaries"
    assert squared_norm(x) == 250_000.0
