# =============================================================================
#  04.04 -- eigh and SVD
# =============================================================================
#
#  np.linalg has two eigensolvers, and the general one is the wrong default.
#
#      np.linalg.eig(a)    any square matrix. Returns COMPLEX eigenvalues in
#                          NO PARTICULAR ORDER, because that is all a general
#                          matrix promises.
#      np.linalg.eigh(a)   symmetric (Hermitian) matrices. Returns REAL
#                          eigenvalues in ASCENDING order and orthonormal
#                          eigenvectors, faster and more accurately, because
#                          symmetry guarantees all of that.
#
#  Covariance matrices, Gram matrices, Laplacians, Hessians of real functions:
#  symmetric, all of them, so `eigh` is the one you want most of the time.
#  Johansson, *Numerical Python* ch. 5, "Eigenvalue Problems", introduces the
#  problem; Kneusel, *Math for Deep Learning* ch. 6, "Symmetric, Orthogonal,
#  and Unitary Matrices" and "Eigenvectors and Eigenvalues", explains why
#  symmetry buys real values and orthogonal vectors.
#
#  The singular value decomposition a = u @ diag(s) @ vt is the tool for
#  non-square matrices, and the source of the best low-rank approximation
#  there is: keep the k largest singular values and their vectors, and the
#  error you make is exactly the norm of the ones you dropped (Eckart-Young).
#  Ask for `full_matrices=False`: the "economy" shapes (m, r) and (r, n) are
#  all you need, and for a tall matrix the full (m, m) u is mostly padding.
#
#  TASK
#    Make `principal_directions` use eigh and return the largest eigenvalue
#    first, and implement `low_rank_approximation` with the economy SVD.
#
#  RUN IT
#    ./npt test 04_04
#
# =============================================================================

import numpy as np


def principal_directions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and matching eigenvectors (as columns) of the
    covariance matrix of the rows of `x`."""
    centred = x - x.mean(axis=0)
    covariance = centred.T @ centred / (x.shape[0] - 1)
    # TODO: the covariance matrix is symmetric. eig returns complex values in
    # whatever order LAPACK found them; use eigh and sort descending.
    values, vectors = np.linalg.eig(covariance)
    return values, vectors


def low_rank_approximation(a: np.ndarray, k: int) -> np.ndarray:
    """The best rank-k approximation of `a` in the least-squares sense."""
    # TODO: keep the k largest singular values. With the default
    # full_matrices=True, u is (m, m) and vt is (n, n), so this expression does
    # not even have compatible shapes for a non-square matrix.
    u, s, vt = np.linalg.svd(a)
    return u @ np.diag(s) @ vt


def test_principal_directions_of_an_elongated_cloud():
    rng = np.random.default_rng(0)
    # Points spread 10x more along (1, 1) than along (1, -1).
    n = 2000
    t = rng.standard_normal(n) * 10.0
    s = rng.standard_normal(n)
    x = np.column_stack([t + s, t - s]) / np.sqrt(2.0)
    values, vectors = principal_directions(x)
    assert values.dtype.kind == "f", "eigh returns real values; eig returns complex"
    assert values[0] > values[1], "largest eigenvalue first"
    assert abs(values[0] / values[1] - 100.0) < 15.0
    first = vectors[:, 0]
    np.testing.assert_allclose(np.abs(first), [1.0, 1.0] / np.sqrt(2.0), atol=0.05)


def test_principal_directions_are_sorted_and_orthonormal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 5)) * np.array([1.0, 5.0, 2.0, 0.5, 3.0])
    values, vectors = principal_directions(x)
    assert np.all(np.diff(values) <= 0), "descending order"
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(5), atol=1e-10)


def test_low_rank_approximation_is_optimal():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((30, 20))
    s = np.linalg.svd(a, compute_uv=False)
    for k in (1, 5, 19):
        approx = low_rank_approximation(a, k)
        assert np.linalg.matrix_rank(approx) == k
        # Eckart-Young: the error of the best rank-k approximation is exactly
        # the norm of the discarded singular values.
        error = np.linalg.norm(a - approx)
        np.testing.assert_allclose(error, np.linalg.norm(s[k:]), rtol=1e-10)


def test_low_rank_approximation_of_a_rank_k_matrix_is_exact():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((40, 3)) @ rng.standard_normal((3, 25))
    np.testing.assert_allclose(low_rank_approximation(a, 3), a, atol=1e-10)
