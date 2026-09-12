# =============================================================================
#  02.06 -- Pairwise distances
# =============================================================================
#
#  Given n points a[i] and m points b[j] in d dimensions, the (n, m) table
#  of squared distances |a[i] - b[j]|^2 is the workhorse of nearest
#  neighbours, clustering, kernels and attention. There are two ways to
#  write it, and knowing both is the point.
#
#  THE BROADCAST WAY. Insert axes (02.02) so the points line up pairwise:
#
#      a[:, None, :] - b[None, :, :]        (n, 1, d) - (1, m, d) -> (n, m, d)
#
#  then square and sum over the LAST axis, the coordinates, to get (n, m).
#  Clear, correct, and it materialises n * m * d numbers. For 10 000 points
#  in 128 dimensions that is 100 GB. It does not fit.
#
#  THE GRAM WAY. Expand the square: |a - b|^2 = |a|^2 + |b|^2 - 2 a.b.
#  The cross term for all pairs at once is one matrix product, a @ b.T,
#  which BLAS runs near memory speed, and the temporaries are (n, 1), (1, m)
#  and (n, m). Kneusel, Math for Deep Learning, ch. 5 covers the identity;
#  Johansson, Numerical Python 3rd ed., ch. 2 "Matrix and Vector
#  Operations" covers the product.
#
#  The price of the Gram way is cancellation. For two nearly equal points
#  the three terms are large and their true difference is tiny, so floating
#  point rounding can leave a result slightly below zero: -1e-12 instead of
#  0. Take the square root of that and you have NaN. Clip at zero. (06.06
#  is about this class of problem.)
#
#  TASK
#    Fix the axis in `sq_distances_broadcast` and the sign problem in
#    `sq_distances_gram`.
#
#  RUN IT
#    ./npt test 02_06
#
# =============================================================================

import numpy as np


def sq_distances_broadcast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d[i, j] = |a[i] - b[j]|^2 via an (n, m, d) difference table."""
    # TODO: the sum must run over the coordinate axis, the last one. Summing
    # over axis 0 adds up the POINTS and leaves an (m, d) result.
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff**2, axis=0)


def sq_distances_gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The same table via |a|^2 + |b|^2 - 2 a.b, using one matrix product."""
    # TODO: rounding can push a near-zero distance a hair below zero, and
    # sqrt of that is NaN. Clip the result at 0.
    a_sq = np.sum(a**2, axis=1)[:, None]
    b_sq = np.sum(b**2, axis=1)[None, :]
    return a_sq + b_sq - 2.0 * (a @ b.T)


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
