# Solution -- 04.04 eigh and SVD

import numpy as np


def principal_directions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and matching eigenvectors (as columns) of the
    covariance matrix of the rows of `x`."""
    centred = x - x.mean(axis=0)
    covariance = centred.T @ centred / (x.shape[0] - 1)
    # eigh: for a symmetric matrix. It returns real values in ASCENDING order
    # and orthonormal eigenvectors, and is faster and more accurate than the
    # general eig, which would also return complex dtypes with zero imaginary
    # parts. Reverse to get the largest first.
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    return values[order], vectors[:, order]


def low_rank_approximation(a: np.ndarray, k: int) -> np.ndarray:
    """The best rank-k approximation of `a` in the least-squares sense."""
    # full_matrices=False gives the "economy" SVD: u is (m, r), vt is (r, n)
    # with r = min(m, n), instead of square m x m and n x n matrices most of
    # which multiply zeros. Keep the k largest singular triplets.
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k]


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
