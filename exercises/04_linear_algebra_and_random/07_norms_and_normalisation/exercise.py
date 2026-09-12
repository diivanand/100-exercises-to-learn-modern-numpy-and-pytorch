# =============================================================================
#  04.07 -- Norms and normalisation
# =============================================================================
#
#  `np.linalg.norm(x)` with no axis flattens x and returns one number. With
#  `axis=1` it returns one number per row, shape (n,). And that is where the
#  trouble starts, because you almost always want to DIVIDE by that norm,
#  and (n, d) / (n,) does not do what it looks like it does.
#
#  Broadcasting (02.01) aligns shapes from the RIGHT. (n, d) against (n,)
#  lines the n up with d. If n != d that is an error, which is annoying but
#  honest. If n == d it broadcasts happily and divides each COLUMN by a row's
#  norm. Square matrices are common enough (Gram matrices, attention scores,
#  any all-pairs quantity) that this bug is common too.
#
#  The fix is `keepdims=True`. The reduced axis stays in the shape as a 1,
#  so the norm is (n, 1), which broadcasts against (n, d) row by row, which
#  is what you meant. Reductions over axis=0, such as the per-column mean
#  and standard deviation used to standardise features, do not have this
#  problem because a (d,) result already aligns with the last axis. The
#  rule of thumb: reducing the LAST axis and broadcasting back needs
#  keepdims; reducing an earlier axis does not.
#
#  Kneusel, *Math for Deep Learning* ch. 6, "Vector Norms and Distance
#  Metrics", defines the norms; the cosine similarity below is the dot
#  product of unit vectors, which is why `unit_rows` does most of the work.
#
#  TASK
#    Fix `unit_rows` so that it works for every shape, including square,
#    and implement `standardise`.
#
#  RUN IT
#    ./npt test 04_07
#
# =============================================================================

import numpy as np


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Each row of `x` scaled to unit Euclidean length."""
    # TODO: `norms` has shape (n,). For a non-square x this raises; for a
    # square x it silently divides the wrong axis. Keep the reduced axis.
    norms = np.linalg.norm(x, axis=1)
    return x / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) matrix of cosines between the rows of `a` (n, d) and of `b` (m, d)."""
    return unit_rows(a) @ unit_rows(b).T


def standardise(x: np.ndarray) -> np.ndarray:
    """Each COLUMN of `x` shifted to mean 0 and scaled to standard deviation 1."""
    # TODO: per-column statistics, i.e. a reduction over axis=0.
    return (x - x.mean()) / x.std()


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
