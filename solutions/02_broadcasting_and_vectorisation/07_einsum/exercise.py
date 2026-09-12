# Solution -- 02.07 einsum
import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each batch index: a[k] @ b[k]. a is (B, n, p), b is (B, p, m)."""
    # Read the subscripts as "for each b, sum over j of a[b,i,j] * b[b,j,k]".
    # The repeated letter j is the one summed; b, i, k survive to the output.
    return np.einsum("bij,bjk->bik", a, b)


def trace_of_each(a: np.ndarray) -> np.ndarray:
    """The trace of each (n, n) matrix in a (B, n, n) stack."""
    # A letter repeated WITHIN one operand walks the diagonal.
    return np.einsum("bii->b", a)


def bilinear(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each row: x[b] . W . y[b], giving (B,)."""
    # Three operands, two contractions. optimize=True lets einsum pick the
    # order of pairwise products (x @ W first, then dot with y) instead of
    # building the full (B, i, j) product; it matters once the sizes grow.
    return np.einsum("bi,ij,bj->b", x, w, y, optimize=True)


def test_batched_matmul_matches_the_operator():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 5, 2))
    np.testing.assert_allclose(batched_matmul(a, b), a @ b)
    assert batched_matmul(a, b).shape == (4, 3, 2)


def test_trace_of_each():
    a = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    np.testing.assert_allclose(trace_of_each(a), [0 + 4 + 8, 9 + 13 + 17])


def test_bilinear_matches_the_loop():
    rng = np.random.default_rng(11)
    x, w, y = rng.normal(size=(6, 3)), rng.normal(size=(3, 4)), rng.normal(size=(6, 4))
    expected = np.array([x[k] @ w @ y[k] for k in range(6)])
    np.testing.assert_allclose(bilinear(x, w, y), expected)
    assert bilinear(x, w, y).shape == (6,)
