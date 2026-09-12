# Solution -- 03.01 Views vs copies
import numpy as np


def first_row(matrix: np.ndarray) -> np.ndarray:
    """The first row, as a view: no pixels are copied."""
    # Basic indexing (integers and slices) always returns a view. Copying
    # here would cost a full row of memory for nothing.
    return matrix[0]


def every_other(xs: np.ndarray) -> np.ndarray:
    """Elements 0, 2, 4, ... as a view."""
    # A stride of two is still a slice, so still a view: the result simply
    # has strides twice as large as xs's (03.02).
    return xs[::2]


def safe_head(xs: np.ndarray, n: int) -> np.ndarray:
    """The first n elements, as an INDEPENDENT array the caller may edit."""
    # A slice is a view; hand one out and the caller's edits land in xs.
    # The contract says independent, so copy. (Core Guidelines F.15 in spirit:
    # say what you mean about ownership.)
    return xs[:n].copy()


def selected(xs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """xs at the given positions."""
    # Integer-array ("fancy") indexing cannot be a view: the positions need
    # not be evenly spaced, so no strides describe them. NumPy copies.
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
