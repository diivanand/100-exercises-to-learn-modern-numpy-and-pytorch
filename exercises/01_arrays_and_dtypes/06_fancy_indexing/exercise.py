# =============================================================================
#  01.06 -- Fancy indexing
# =============================================================================
#
#  Index an array with an INTEGER ARRAY (or list) and you get the elements at
#  those positions, in that order, repeats allowed: `xs[[2, 0, 2]]`. NumPy
#  calls this fancy indexing (Johansson, ch. 2, "Fancy Indexing and
#  Boolean-Valued Indexing"; McKinney calls it the same). Unlike a slice it
#  always COPIES -- there is no stride that describes "elements 2, 0, 2".
#
#  With two index arrays on a 2-D array the rule surprises almost everyone:
#
#      m[[0, 2], [1, 3]]       # the PAIRS (0,1) and (2,3): a 1-D result
#
#  The index arrays are broadcast together (chapter 02) and each resulting
#  pair picks one element. That is the right behaviour for "give me these
#  coordinates". It is the wrong behaviour for "give me rows 0 and 2,
#  columns 1 and 3 -- the 2 x 2 block". For the block you want the row
#  indices as a column and the column indices as a row, so that
#  broadcasting pairs every row with every column:
#
#      m[np.ix_([0, 2], [1, 3])]           # shape (2, 2)
#      m[[0, 2]][:, [1, 3]]                # also works: two copies
#
#  Mixing a slice with an index array is fine and common: `m[:, [3, 1]]`
#  reorders columns; `m[order]` with `order = np.argsort(key)` sorts rows by
#  a key. `np.take(a, idx, axis=)` is the function form.
#
#  TASK
#    Fix `block`, then write `top_k` using argsort and an index array.
#
#  RUN IT
#    ./npt test 01_06
#
# =============================================================================

import numpy as np


def block(m: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    """The sub-matrix at the given rows and columns, shape (len(rows), len(cols))."""
    # TODO: this selects the PAIRS (rows[i], cols[i]), which is what `pick`
    # below is for. Use np.ix_ so the two lists broadcast into a grid.
    return m[rows, cols]


def pick(m: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    """The elements m[rows[i], cols[i]], one per i."""
    return m[rows, cols]


def top_k(xs: np.ndarray, k: int) -> np.ndarray:
    """The `k` largest values of `xs`, largest first."""
    # TODO: np.argsort gives the ascending order of positions; index with
    # the last k of them, reversed.
    raise NotImplementedError


def test_block_is_a_sub_matrix():
    m = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(block(m, [0, 2], [1, 3]), [[1, 3], [9, 11]])
    assert block(m, [3], [0, 1, 2]).shape == (1, 3)


def test_pick_is_pairs():
    m = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(pick(m, [0, 2], [1, 3]), [1, 11])


def test_fancy_indexing_copies():
    m = np.arange(16).reshape(4, 4)
    b = block(m, [0, 1], [0, 1])
    assert not np.shares_memory(b, m)
    b[:] = -1
    assert m[0, 0] == 0


def test_top_k():
    xs = np.array([3.0, 9.0, 1.0, 7.0, 5.0])
    np.testing.assert_array_equal(top_k(xs, 3), [9.0, 7.0, 5.0])
    np.testing.assert_array_equal(top_k(xs, 1), [9.0])


def test_index_arrays_may_repeat_and_reorder():
    xs = np.array([10, 20, 30])
    np.testing.assert_array_equal(xs[[2, 0, 2, 2]], [30, 10, 30, 30])
