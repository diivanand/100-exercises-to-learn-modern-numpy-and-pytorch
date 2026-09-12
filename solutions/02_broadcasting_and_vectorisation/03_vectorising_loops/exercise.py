# Solution -- 02.03 Vectorising loops
import time

import numpy as np


def count_close_pairs(xs: np.ndarray, tol: float) -> int:
    """How many pairs i < j have |xs[i] - xs[j]| < tol."""
    # Every pairwise difference at once as an (n, n) table (02.02), then a
    # boolean mask, then keep the strict upper triangle so each pair counts
    # once. n = 2000 is a 4 M-element table: fine. n = 100 000 would not be;
    # at that point sort first and use searchsorted (05.01).
    close = np.abs(xs[:, None] - xs[None, :]) < tol
    return int(np.triu(close, k=1).sum())


def min_max_normalise(matrix: np.ndarray) -> np.ndarray:
    """Rescale each column into [0, 1]; a constant column becomes all zeros."""
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    span = hi - lo
    # Where the span is zero, divide by 1 instead: the numerator is zero
    # there anyway. np.where evaluates both branches (02.05), so guard the
    # divisor rather than the result.
    safe_span = np.where(span == 0, 1.0, span)
    return (matrix - lo) / safe_span


# --- the slow references the tests compare against ---------------------------


def _count_close_pairs_loop(xs: np.ndarray, tol: float) -> int:
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if abs(xs[i] - xs[j]) < tol:
                count += 1
    return count


def _min_max_normalise_loop(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    out = np.empty_like(matrix, dtype=np.float64)
    for j in range(cols):
        lo = min(matrix[i, j] for i in range(rows))
        hi = max(matrix[i, j] for i in range(rows))
        for i in range(rows):
            out[i, j] = 0.0 if hi == lo else (matrix[i, j] - lo) / (hi - lo)
    return out


def _best_of(fn, repeats: int = 3) -> float:
    fn()  # warm-up (03.06 explains why)
    return min(_timed(fn) for _ in range(repeats))


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_count_close_pairs_matches_the_loop():
    rng = np.random.default_rng(1)
    xs = rng.normal(size=400)
    assert count_close_pairs(xs, 0.05) == _count_close_pairs_loop(xs, 0.05)
    assert count_close_pairs(xs, 0.0) == 0
    assert count_close_pairs(np.zeros(4), 1e-9) == 6


def test_count_close_pairs_is_vectorised():
    rng = np.random.default_rng(2)
    xs = rng.normal(size=600)
    loop = _best_of(lambda: _count_close_pairs_loop(xs, 0.1))
    fast = _best_of(lambda: count_close_pairs(xs, 0.1))
    assert fast * 5 < loop, f"loop {loop:.4f}s vs yours {fast:.4f}s"


def test_min_max_normalise_matches_the_loop():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(50, 6)) * 10
    m[:, 2] = 4.0  # a constant column
    out = min_max_normalise(m)
    np.testing.assert_allclose(out, _min_max_normalise_loop(m))
    np.testing.assert_allclose(out.min(axis=0), [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(out.max(axis=0), [1, 1, 0, 1, 1, 1])


def test_min_max_normalise_is_vectorised():
    rng = np.random.default_rng(4)
    m = rng.normal(size=(2000, 8))
    loop = _best_of(lambda: _min_max_normalise_loop(m))
    fast = _best_of(lambda: min_max_normalise(m))
    assert fast * 5 < loop, f"loop {loop:.4f}s vs yours {fast:.4f}s"
