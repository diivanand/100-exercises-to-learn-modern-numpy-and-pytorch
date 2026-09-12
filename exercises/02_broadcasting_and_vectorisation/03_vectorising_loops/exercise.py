# =============================================================================
#  02.03 -- Vectorising loops
# =============================================================================
#
#  A Python `for` loop over the elements of an array costs about a
#  microsecond per iteration: the interpreter fetches an element, boxes it
#  into a Python float, dispatches the operator, and unboxes the result. The
#  same operation applied to the whole array in one NumPy call costs a few
#  nanoseconds per element, because the loop runs in compiled C over a
#  contiguous buffer, and often uses SIMD instructions to do several
#  elements per cycle. That gap, two to three orders of magnitude, is what
#  "vectorised" means in NumPy, and it is the single most important
#  performance idea in the library. Johansson, Numerical Python 3rd ed.,
#  ch. 2 "Vectorized Expressions" puts it as: express the computation in
#  terms of whole-array operations, and let NumPy loop.
#
#  The habit to build is to see a loop and ask what array operation it is:
#
#      for i: for j: d[i, j] = f(x[i], y[j])   ->   f(x[:, None], y[None, :])
#      for j: col = m[:, j]; ...              ->   m.min(axis=0), m.max(axis=0)
#
#  The pairwise case builds an (n, n) table (02.02). That is fine for
#  thousands of elements and hopeless for millions, at which point the
#  answer is a different algorithm (sort and search, 05.01), not a faster
#  loop. Vectorising trades memory for time; know which one you have.
#
#  The tests time your version against the loop it replaces. The loop
#  reference is kept small so the test finishes quickly, and the bar is a
#  fifth of the speed-up you should see, so it is not a flaky benchmark.
#
#  TASK
#    Rewrite the two functions without a Python loop over elements.
#
#  RUN IT
#    ./npt test 02_03
#
# =============================================================================

import time

import numpy as np


def count_close_pairs(xs: np.ndarray, tol: float) -> int:
    """How many pairs i < j have |xs[i] - xs[j]| < tol."""
    # TODO: build the whole (n, n) table of differences with broadcasting,
    # turn it into a boolean mask, and count the strict upper triangle
    # (np.triu with k=1) so each pair is counted once.
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if abs(xs[i] - xs[j]) < tol:
                count += 1
    return count


def min_max_normalise(matrix: np.ndarray) -> np.ndarray:
    """Rescale each column into [0, 1]; a constant column becomes all zeros."""
    # TODO: min and max take an axis. The division needs a guard for
    # constant columns; np.where (02.05) can supply a safe divisor.
    rows, cols = matrix.shape
    out = np.empty_like(matrix, dtype=np.float64)
    for j in range(cols):
        lo = min(matrix[i, j] for i in range(rows))
        hi = max(matrix[i, j] for i in range(rows))
        for i in range(rows):
            out[i, j] = 0.0 if hi == lo else (matrix[i, j] - lo) / (hi - lo)
    return out


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
