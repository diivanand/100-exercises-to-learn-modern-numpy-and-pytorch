# =============================================================================
#  02.08 -- np.vectorize is a loop
# =============================================================================
#
#  `np.vectorize(f)` takes a Python function of scalars and returns one that
#  accepts arrays. It looks like the thing 02.03 was about. It is not: the
#  documentation says so in its first paragraph -- "The vectorize function is
#  provided primarily for convenience, not for performance. The implementation
#  is essentially a for loop." Every element still goes through the
#  interpreter, at a microsecond each. `np.apply_along_axis(f, axis, m)` is
#  the same story one level up: it calls f once per row (or column) from
#  Python.
#
#  Both have a place: a genuinely scalar function from some other library
#  that you need to map once over a small array. Neither has a place in
#  code you care about the speed of. The fix is always the same as in
#  02.03: find the array operation. `x if x >= 0 else s * x` is
#  `np.where(x >= 0, x, s * x)`; `np.linalg.norm` on each row is
#  `np.linalg.norm(m, axis=1)`, because nearly every NumPy reduction takes
#  an axis (01.07).
#
#  np.vectorize has a second, nastier property: unless you tell it the
#  output type, it calls f on the FIRST element to guess the dtype of the
#  whole result. A function that returns the input unchanged for positive
#  numbers and a scaled float for negative ones therefore returns an
#  INTEGER array when the first element is a positive int, and truncates
#  every negative output to 0. The tests check for exactly that.
#
#  TASK
#    Rewrite `leaky_relu` and `row_norms` with array operations.
#
#  RUN IT
#    ./npt test 02_08
#
# =============================================================================

import time

import numpy as np


def leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    """x where x >= 0, slope * x elsewhere."""
    # TODO: this is a Python loop in disguise, and its output dtype depends
    # on the first element. Use np.where.
    return np.vectorize(lambda v: v if v >= 0 else slope * v)(x)


def row_norms(matrix: np.ndarray) -> np.ndarray:
    """The Euclidean length of each row."""
    # TODO: one Python call per row. np.linalg.norm takes an axis.
    return np.apply_along_axis(np.linalg.norm, 1, matrix)


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
