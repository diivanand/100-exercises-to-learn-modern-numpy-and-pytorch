# Solution -- 04.01 Matrix multiplication and @

import numpy as np


def linear_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = x @ w.T + b for a batch of rows in `x` (n, d_in) -> (n, d_out)."""
    # `@` is matrix multiplication; `*` is elementwise. w is stored as
    # (d_out, d_in), the way most frameworks store weights, so it is w.T that
    # has the shape the product needs.
    return x @ w.T + b


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply stacks of matrices: (..., n, k) @ (..., k, m) -> (..., n, m)."""
    # `@` treats the last two axes as the matrix and broadcasts everything in
    # front of them, so a stack of 8 matrices times a single shared matrix is
    # one call, not a loop.
    return a @ b


def apply_to_vectors(m: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Apply one matrix (n, k) to a stack of vectors (..., k) -> (..., n)."""
    # np.matvec (NumPy 2.2) says exactly this: the last axis of `vs` is the
    # vector. The older spelling is `vs @ m.T`, which works but says something
    # about transposes rather than about intent.
    return np.matvec(m, vs)


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
