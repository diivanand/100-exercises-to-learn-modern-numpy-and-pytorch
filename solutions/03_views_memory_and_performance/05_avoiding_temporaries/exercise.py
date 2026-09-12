# Solution -- 03.05 Avoiding temporaries
import tracemalloc

import numpy as np


def affine_into(x: np.ndarray, a: float, b: float, out: np.ndarray) -> np.ndarray:
    """out = a * x + b, written into `out` with no temporary array."""
    # Two ufunc calls, both writing into `out`. The expression `a * x + b`
    # would allocate one full-size array for `a * x` and another for the sum.
    np.multiply(x, a, out=out)
    np.add(out, b, out=out)
    return out


def squared_norm(x: np.ndarray) -> float:
    """sum(x * x) without building x * x."""
    # `np.sum(x**2)` materialises x**2 first. A dot product streams through
    # x once and keeps a running total; BLAS does it for float arrays.
    return float(np.dot(x, x))


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
