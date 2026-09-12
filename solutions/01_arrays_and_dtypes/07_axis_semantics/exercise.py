# Solution -- 01.07 Axis semantics
import numpy as np


def row_means(m: np.ndarray) -> np.ndarray:
    """One mean per row of a 2-D array: shape (rows,)."""
    # axis=1 is the axis that DISAPPEARS: reduce along the columns of each
    # row, leaving one value per row.
    return m.mean(axis=1)


def normalise_rows(m: np.ndarray) -> np.ndarray:
    """Scale each row so that it sums to 1."""
    # keepdims keeps the reduced axis as length 1, so the sums have shape
    # (rows, 1) and broadcast down each row instead of across the columns.
    return m / m.sum(axis=1, keepdims=True)


def standardise_columns(m: np.ndarray) -> np.ndarray:
    """Each column with its own mean subtracted and divided by its own std."""
    return (m - m.mean(axis=0)) / m.std(axis=0)


def test_row_means():
    m = np.array([[1.0, 3.0], [10.0, 20.0], [0.0, 0.0]])
    np.testing.assert_allclose(row_means(m), [2.0, 15.0, 0.0])


def test_normalise_rows_on_a_non_square_matrix():
    m = np.array([[1.0, 1.0, 2.0], [3.0, 1.0, 0.0]])
    out = normalise_rows(m)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(out[0], [0.25, 0.25, 0.5])


def test_normalise_rows_on_a_square_matrix():
    # On a square matrix a wrong broadcast direction does not raise, it just
    # divides the wrong thing. This test exists to catch exactly that.
    m = np.array([[1.0, 3.0], [8.0, 8.0]])
    np.testing.assert_allclose(normalise_rows(m), [[0.25, 0.75], [0.5, 0.5]])


def test_standardise_columns():
    rng = np.random.default_rng(1)
    m = rng.normal(loc=[5.0, -2.0, 100.0], scale=[1.0, 3.0, 10.0], size=(1000, 3))
    z = standardise_columns(m)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1.0)


def test_the_reduced_axis_disappears():
    m = np.ones((2, 3, 4))
    assert m.sum(axis=0).shape == (3, 4)
    assert m.sum(axis=1).shape == (2, 4)
    assert m.sum(axis=2).shape == (2, 3)
    assert m.sum(axis=(1, 2)).shape == (2,)
    assert m.sum(axis=1, keepdims=True).shape == (2, 1, 4)
