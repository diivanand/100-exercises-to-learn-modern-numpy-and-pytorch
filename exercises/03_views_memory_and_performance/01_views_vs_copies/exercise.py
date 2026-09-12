# =============================================================================
#  03.01 -- Views vs copies
# =============================================================================
#
#  An ndarray is a small object that points at a block of memory and
#  describes how to read it: a shape, a dtype, and strides (03.02). Two
#  arrays can point at the SAME block with different descriptions. That is
#  a view, and NumPy hands them out constantly, because they are free:
#
#      a[1:5], a[::2], a[0]      slicing and integer indexing
#      a.T, a.reshape(...)       when the layout allows (03.03)
#      a[:, None]                adding axes (02.02)
#
#  Johansson, Numerical Python 3rd ed., ch. 2 "Views": "for efficiency,
#  NumPy strives to create views rather than copies", and then the warning
#  that follows from it: modify the view and you have modified the original.
#  Sometimes that is exactly what you want (03.04). Sometimes it is a bug
#  that surfaces three functions away, when a caller's data changes because
#  a helper returned a slice and someone wrote into it.
#
#  The operations that CANNOT be views copy: integer-array ("fancy")
#  indexing and boolean masks, because the selected positions have no
#  regular stride; and any arithmetic, because the result is new data.
#
#  Three ways to find out what you are holding:
#
#      np.shares_memory(a, b)    do they overlap?  (the reliable question)
#      a.base                    the array this one views, or None
#      a.flags.owndata           does a own its block?
#
#  and one rule for writing functions: say what you return. A function that
#  promises the caller an independent array must `.copy()`. A function that
#  promises not to copy a large matrix must not. Both are contracts, and the
#  tests below hold you to each.
#
#  TASK
#    Fix `first_row` and `safe_head` so that each returns what it promises.
#
#  RUN IT
#    ./npt test 03_01
#
# =============================================================================

import numpy as np


def first_row(matrix: np.ndarray) -> np.ndarray:
    """The first row, as a view: no pixels are copied."""
    # TODO: this copies a full row for nothing. Basic indexing already gives
    # a view.
    return matrix[0].copy()


def every_other(xs: np.ndarray) -> np.ndarray:
    """Elements 0, 2, 4, ... as a view."""
    return xs[::2]


def safe_head(xs: np.ndarray, n: int) -> np.ndarray:
    """The first n elements, as an INDEPENDENT array the caller may edit."""
    # TODO: a slice is a view, so the caller's edits land in xs. The contract
    # says independent.
    return xs[:n]


def selected(xs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """xs at the given positions."""
    return xs[indices]


def test_first_row_is_a_view():
    m = np.arange(12.0).reshape(3, 4).copy()  # .copy() so m owns its data
    row = first_row(m)
    assert np.shares_memory(row, m)
    assert row.base is m
    np.testing.assert_array_equal(row, [0.0, 1.0, 2.0, 3.0])


def test_every_other_is_a_view():
    xs = np.arange(10)
    view = every_other(xs)
    np.testing.assert_array_equal(view, [0, 2, 4, 6, 8])
    assert np.shares_memory(view, xs)
    view[0] = 99
    assert xs[0] == 99


def test_safe_head_is_independent():
    xs = np.arange(5.0)
    head = safe_head(xs, 3)
    np.testing.assert_array_equal(head, [0.0, 1.0, 2.0])
    head[:] = -1.0
    np.testing.assert_array_equal(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert not np.shares_memory(head, xs)


def test_selected_is_a_copy():
    xs = np.arange(6) * 10
    picked = selected(xs, np.array([5, 0, 3]))
    np.testing.assert_array_equal(picked, [50, 0, 30])
    assert not np.shares_memory(picked, xs)


def test_what_owns_what():
    # Not a task. The three flags that tell you what you are holding.
    base = np.zeros((4, 4))
    view = base[1:3]
    copy = base[1:3].copy()
    assert base.flags.owndata and not view.flags.owndata and copy.flags.owndata
    assert view.base is base and copy.base is None
    assert np.shares_memory(view, base) and not np.shares_memory(copy, base)
