# =============================================================================
#  02.07 -- einsum
# =============================================================================
#
#  `np.einsum` is one function that covers matrix products, batched
#  products, traces, transposes, outer products and most contractions you
#  will ever write, from a string that says which axes go where:
#
#      np.einsum("ij,jk->ik", a, b)     a @ b
#      np.einsum("bij,bjk->bik", a, b)  a @ b for every batch index b
#      np.einsum("ii->i", a)            the diagonal
#      np.einsum("ii->", a)             the trace
#      np.einsum("ij->ji", a)           the transpose
#      np.einsum("i,j->ij", x, y)       the outer product
#
#  Read the string as a loop: each letter is an index; a letter that appears
#  in an input but not after the arrow is summed over; a letter repeated
#  WITHIN one operand walks its diagonal. Johansson, Numerical Python 3rd
#  ed., ch. 2 "Matrix and Vector Operations" introduces it after np.outer
#  and np.kron; the same function exists in PyTorch as torch.einsum (07.07)
#  with the same strings, which is a good reason to be fluent.
#
#  Two practical notes. The letters are yours to choose, so choose ones that
#  name the axes ("bhqk" for batch, head, query, key) and the string becomes
#  documentation. And for three or more operands pass `optimize=True`: by
#  default einsum contracts everything in one pass, which for
#  "bi,ij,bj->b" means building the full (B, i, j) product; with
#  optimisation it does x @ W first and then a row-wise dot, which is both
#  faster and far smaller.
#
#  TASK
#    Fix the subscripts in `batched_matmul`, and make `bilinear` produce
#    one number per row.
#
#  RUN IT
#    ./npt test 02_07
#
# =============================================================================

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each batch index: a[k] @ b[k]. a is (B, n, p), b is (B, p, m)."""
    # TODO: the contracted index must be a's LAST axis and b's FIRST
    # non-batch axis. As written this multiplies a by b transposed.
    return np.einsum("bij,bkj->bik", a, b)


def trace_of_each(a: np.ndarray) -> np.ndarray:
    """The trace of each (n, n) matrix in a (B, n, n) stack."""
    return np.einsum("bii->b", a)


def bilinear(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each row: x[b] . W . y[b], giving (B,)."""
    # TODO: keeping i and j after the arrow leaves a (B, i, j) table of
    # products instead of summing them. Only b should survive; and with three
    # operands, ask einsum to optimise the contraction order.
    return np.einsum("bi,ij,bj->bij", x, w, y)


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
