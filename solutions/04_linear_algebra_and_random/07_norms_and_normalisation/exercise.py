# Solution -- 04.07 Norms and normalisation

import numpy as np


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Each row of `x` scaled to unit Euclidean length."""
    # keepdims=True keeps the norm as an (n, 1) column, which broadcasts
    # against (n, d) the way we mean: one divisor per ROW. Without it the
    # norm is (n,), which broadcasts against the last axis -- one divisor per
    # COLUMN -- and only raises an error when n != d.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) matrix of cosines between the rows of `a` (n, d) and of `b` (m, d)."""
    return unit_rows(a) @ unit_rows(b).T


def standardise(x: np.ndarray) -> np.ndarray:
    """Each COLUMN of `x` shifted to mean 0 and scaled to standard deviation 1."""
    # Reductions over axis=0 give one number per column, shape (d,), which is
    # the right shape to broadcast against rows. keepdims is not needed here,
    # but it would not hurt either: (1, d) broadcasts the same way.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std


def test_unit_rows_have_unit_norm():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3)) * 10.0
    u = unit_rows(x)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), np.ones(5))
    # Direction is preserved: each unit row is a positive multiple of its source.
    for row, unit in zip(x, u, strict=True):
        np.testing.assert_allclose(row / np.linalg.norm(row), unit)


def test_unit_rows_on_a_square_matrix():
    # The trap. For a square matrix, dividing by an (n,) norm broadcasts along
    # the wrong axis without complaint; the rows end up with the wrong
    # lengths and nothing raises.
    x = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    u = unit_rows(x)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), np.ones(3))
    np.testing.assert_allclose(u[0], [0.6, 0.8, 0.0])


def test_cosine_similarity():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 5.0], [-1.0, 0.0]])
    expected = np.array([[1.0, 0.0, -1.0], [np.sqrt(0.5), np.sqrt(0.5), -np.sqrt(0.5)]])
    np.testing.assert_allclose(cosine_similarity(a, b), expected)


def test_standardise_columns():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 4)) * np.array([1.0, 10.0, 0.1, 100.0]) + 5.0
    z = standardise(x)
    np.testing.assert_allclose(z.mean(axis=0), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), np.ones(4), atol=1e-12)
