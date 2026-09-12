# =============================================================================
#  04.01 -- Matrix multiplication and @
# =============================================================================
#
#  In NumPy `*` is elementwise. Two arrays of the same shape multiply
#  position by position, and two arrays of different shapes broadcast (02.01).
#  Neither of those is a matrix product. The matrix product is `@`, or the
#  function `np.matmul`, and it is worth being pedantic about the difference
#  because for square operands `a * b` and `a @ b` have the same shape and
#  different values, so nothing will warn you.
#
#  Johansson, *Numerical Python* ch. 2, "Matrix and Vector Operations", walks
#  through np.dot, np.inner, np.outer, np.tensordot and np.einsum, and notes
#  that `@` "has recently been introduced" and "is still not widely used".
#  That was true in 2015. In 2026 it is the spelling to use for anything
#  two-dimensional or more: it reads as maths, and unlike np.dot it has a
#  single, predictable rule for stacks of matrices:
#
#      the last two axes are the matrix; everything in front broadcasts.
#
#  So (8, 3, 2) @ (2, 5) is eight products with one shared right-hand side,
#  giving (8, 3, 5), and (4, 2, 3) @ (4, 3, 2) pairs the stacks up, giving
#  (4, 2, 2). No loop, and no np.einsum string to get wrong.
#
#  For "one matrix applied to many vectors" NumPy 2.2 added np.matvec, whose
#  signature (n, k) x (..., k) -> (..., n) states the intent directly; the
#  older `vs @ m.T` gives the same numbers. (np.vecmat and np.vecdot are its
#  siblings.)
#
#  Kneusel, *Math for Deep Learning* ch. 5, "Matrix Multiplication in NumPy",
#  covers the same ground from the linear-algebra side, including why the
#  inner dimensions have to agree.
#
#  TASK
#    Fix the three functions so they compute the products their docstrings
#    describe, without loops.
#
#  RUN IT
#    ./npt test 04_01
#
# =============================================================================

import numpy as np


def linear_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = x @ w.T + b for a batch of rows in `x` (n, d_in) -> (n, d_out)."""
    # TODO: this is the elementwise product, broadcast. It happens to run
    # without error whenever the shapes line up, and produces nonsense.
    return x * w.T + b


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply stacks of matrices: (..., n, k) @ (..., k, m) -> (..., n, m)."""
    # TODO: a loop over the leading axis handles (8, 3, 2) @ (2, 5) but not
    # (4, 2, 3) @ (4, 3, 2), and neither is needed.
    return np.stack([ai @ b for ai in a])


def apply_to_vectors(m: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Apply one matrix (n, k) to a stack of vectors (..., k) -> (..., n)."""
    # TODO: `m @ vs` treats `vs` as a matrix (k, ?) and only works by accident
    # when the stack happens to be square. Use np.matvec.
    return m @ vs


def test_linear_layer_is_a_matrix_product():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3))
    w = rng.standard_normal((4, 3))
    b = rng.standard_normal(4)
    expected = np.array([[xi @ wj + bj for wj, bj in zip(w, b, strict=True)] for xi in x])
    np.testing.assert_allclose(linear_layer(x, w, b), expected)
    assert linear_layer(x, w, b).shape == (5, 4)


def test_batched_matmul_broadcasts_over_leading_axes():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 3, 2))
    b = rng.standard_normal((2, 5))  # one matrix shared by all eight
    out = batched_matmul(a, b)
    assert out.shape == (8, 3, 5)
    for i in range(8):
        np.testing.assert_allclose(out[i], a[i] @ b)


def test_batched_matmul_pairs_up_two_stacks():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 2, 3))
    b = rng.standard_normal((4, 3, 2))
    out = batched_matmul(a, b)
    assert out.shape == (4, 2, 2)
    np.testing.assert_allclose(out[3], a[3] @ b[3])


def test_apply_to_vectors():
    m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    vs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = apply_to_vectors(m, vs)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], m[:, 0])
    np.testing.assert_allclose(out[1], m[:, 1])
    np.testing.assert_allclose(out[2], m.sum(axis=1))


def test_elementwise_and_matrix_products_are_different():
    # The shortcut this exercise is about. For a square matrix `x * w.T` has
    # the right SHAPE and the wrong VALUES, so a test on shapes alone would
    # never catch it.
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.zeros(2)
    np.testing.assert_allclose(linear_layer(x, w, b), np.array([[1.0, 3.0], [3.0, 7.0]]))
