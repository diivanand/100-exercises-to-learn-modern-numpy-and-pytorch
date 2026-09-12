# Solution -- 02.06 Pairwise distances
import numpy as np


def sq_distances_broadcast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d[i, j] = |a[i] - b[j]|^2 via an (n, m, d) difference table."""
    # (n, 1, d) - (1, m, d) -> (n, m, d): every pair of rows, all coordinates.
    # Sum over the LAST axis (the coordinates) to get (n, m). Clear, and
    # n * m * d floats of temporary memory.
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff**2, axis=-1)


def sq_distances_gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The same table via |a|^2 + |b|^2 - 2 a.b, using one matrix product."""
    # Expanding |a - b|^2 turns the (n, m, d) temporary into an (n, m)
    # matrix product, which BLAS does at memory speed. The price is
    # cancellation: for two nearly-equal points the three terms are large
    # and their difference is tiny, so rounding can push it a hair below
    # zero. Clip, or sqrt hands back NaN. (06.06 is about this kind of thing.)
    a_sq = np.sum(a**2, axis=1)[:, None]
    b_sq = np.sum(b**2, axis=1)[None, :]
    d = a_sq + b_sq - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of a, the index of the closest row of b."""
    return np.argmin(sq_distances_gram(a, b), axis=1)


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((len(a), len(b)))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            out[i, j] = np.sum((p - q) ** 2)
    return out


def test_broadcast_version_matches_the_loop():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    np.testing.assert_allclose(sq_distances_broadcast(a, b), _reference(a, b))


def test_broadcast_version_shape_is_n_by_m():
    a, b = np.zeros((4, 8)), np.zeros((9, 8))
    assert sq_distances_broadcast(a, b).shape == (4, 9)


def test_gram_version_matches_the_broadcast_version():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(30, 16)), rng.normal(size=(20, 16))
    np.testing.assert_allclose(
        sq_distances_gram(a, b), sq_distances_broadcast(a, b), atol=1e-9
    )


def test_gram_version_never_goes_negative():
    # Points that are exactly equal, and points that are equal up to one ulp,
    # are where the identity loses to rounding.
    rng = np.random.default_rng(8)
    a = rng.normal(size=(50, 64)) * 1e3
    b = a.copy()
    b[::2] = np.nextafter(b[::2], np.inf)
    d = sq_distances_gram(a, b)
    assert np.all(d >= 0.0)
    assert np.all(np.isfinite(np.sqrt(d)))


def test_nearest():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[9.0, 9.0], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(nearest(a, b), [2, 0])
