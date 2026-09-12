# =============================================================================
#  01.07 -- Axis semantics
# =============================================================================
#
#  Reductions -- `sum`, `mean`, `max`, `std`, `argmax` -- take an `axis`
#  argument, and the rule is short: THE AXIS YOU NAME IS THE ONE THAT
#  DISAPPEARS. For a (rows, cols) matrix, `m.sum(axis=0)` collapses the rows
#  and leaves one number per column, shape (cols,); `m.sum(axis=1)` collapses
#  the columns and leaves one per row, shape (rows,). With no axis the whole
#  array collapses to a scalar. Johansson, ch. 2, "Aggregate Functions".
#
#  People remember this backwards because "axis=0 is the row axis" sounds
#  like "sum the rows". Think of the shape instead: cross out the entry at
#  position `axis`, and what is left is the shape of the result.
#
#  `keepdims=True` keeps the crossed-out axis as length 1. That matters when
#  the reduced value is about to be broadcast back against the original:
#  row sums of shape (rows, 1) line up with the rows; row sums of shape
#  (rows,) line up with the COLUMNS, which either raises (if rows != cols)
#  or -- worse -- silently divides the wrong thing (if rows == cols).
#  Chapter 02 is about broadcasting; this is the first place it shows up.
#
#  TASK
#    Fix `row_means` and `normalise_rows`, then write `standardise_columns`.
#
#  RUN IT
#    ./npt test 01_07
#
# =============================================================================

import numpy as np


def row_means(m: np.ndarray) -> np.ndarray:
    """One mean per row of a 2-D array: shape (rows,)."""
    # TODO: axis=0 collapses the rows and leaves one mean per COLUMN.
    return m.mean(axis=0)


def normalise_rows(m: np.ndarray) -> np.ndarray:
    """Scale each row so that it sums to 1."""
    # TODO: the sums have shape (rows,), which broadcasts along the last
    # axis -- the columns. Keep the reduced axis so they line up with rows.
    return m / m.sum(axis=1)


def standardise_columns(m: np.ndarray) -> np.ndarray:
    """Each column with its own mean subtracted and divided by its own std."""
    # TODO
    raise NotImplementedError


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
