# Solution -- 02.08 np.vectorize is a loop
import time

import numpy as np


def leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    """x where x >= 0, slope * x elsewhere."""
    # One np.where over the whole array. The result is float whatever the
    # first element happens to be, which the vectorized version could not
    # promise (it guessed the output dtype from element zero).
    return np.where(x >= 0, x, slope * x)


def row_norms(matrix: np.ndarray) -> np.ndarray:
    """The Euclidean length of each row."""
    # np.linalg.norm takes an axis. apply_along_axis would call it once per
    # row from Python, which is the loop we are trying to avoid.
    return np.linalg.norm(matrix, axis=1)


# --- the convenience versions the tests compare against ----------------------

_leaky_relu_vectorized = np.vectorize(lambda v, slope=0.01: v if v >= 0 else slope * v)


def _row_norms_apply(matrix: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(np.linalg.norm, 1, matrix)


def _best_of(fn, repeats: int = 3) -> float:
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_leaky_relu_values():
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(leaky_relu(x), [-0.02, 0.0, 3.0])
    np.testing.assert_allclose(leaky_relu(x, slope=0.5), [-1.0, 0.0, 3.0])


def test_leaky_relu_dtype_does_not_depend_on_the_first_element():
    # np.vectorize infers the output dtype from the first call. With a
    # negative first element the lambda returns a float; with a positive
    # integer first element it returns an int, and the whole output is int.
    assert leaky_relu(np.array([3, -4])).dtype.kind == "f"
    np.testing.assert_allclose(leaky_relu(np.array([3, -4])), [3.0, -0.04])


def test_leaky_relu_is_faster_than_vectorize():
    x = np.random.default_rng(12).normal(size=100_000)
    slow = _best_of(lambda: _leaky_relu_vectorized(x))
    fast = _best_of(lambda: leaky_relu(x))
    assert fast * 5 < slow, f"np.vectorize {slow:.4f}s vs yours {fast:.4f}s"


def test_row_norms_values():
    m = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(row_norms(m), [5.0, 0.0, np.sqrt(2.0)])


def test_row_norms_is_faster_than_apply_along_axis():
    m = np.random.default_rng(13).normal(size=(20_000, 8))
    slow = _best_of(lambda: _row_norms_apply(m))
    fast = _best_of(lambda: row_norms(m))
    np.testing.assert_allclose(row_norms(m), _row_norms_apply(m))
    assert fast * 5 < slow, f"apply_along_axis {slow:.4f}s vs yours {fast:.4f}s"
