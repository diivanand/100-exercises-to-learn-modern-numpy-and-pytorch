# Solution -- 01.04 Slicing is a view
import numpy as np


def window(xs: np.ndarray, start: int, stop: int) -> np.ndarray:
    """An independent copy of `xs[start:stop]` the caller may modify freely."""
    # The slice is a view; the copy is what makes the promise true.
    return xs[start:stop].copy()


def reverse_in_place(xs: np.ndarray) -> None:
    """Reverse `xs` in place, so the caller's array is reversed."""
    # `xs = xs[::-1]` would rebind the local name to a new view and leave the
    # caller's data untouched. Assigning INTO the slice `xs[:]` writes to the
    # existing memory. NumPy detects that source and destination overlap and
    # copies first, so this is safe.
    xs[:] = xs[::-1]


def every_other_row(m: np.ndarray) -> np.ndarray:
    """Rows 0, 2, 4, ... of `m`, as a view (no copy)."""
    return m[::2]


def test_basic_slices_are_views():
    xs = np.arange(10)
    middle = xs[3:7]
    assert np.shares_memory(middle, xs)
    middle[0] = 99
    assert xs[3] == 99
    # Even a reversed or strided slice is a view: only the strides change.
    assert np.shares_memory(xs[::-1], xs)
    assert np.shares_memory(xs[::3], xs)


def test_window_is_safe_to_modify():
    xs = np.arange(10.0)
    w = window(xs, 2, 5)
    np.testing.assert_array_equal(w, [2.0, 3.0, 4.0])
    w[:] = -1.0
    np.testing.assert_array_equal(xs, np.arange(10.0))


def test_reverse_in_place_changes_the_callers_array():
    xs = np.array([1, 2, 3, 4])
    result = reverse_in_place(xs)
    assert result is None
    np.testing.assert_array_equal(xs, [4, 3, 2, 1])


def test_every_other_row_is_a_view():
    m = np.arange(20).reshape(5, 4)
    rows = every_other_row(m)
    assert rows.shape == (3, 4)
    np.testing.assert_array_equal(rows[1], [8, 9, 10, 11])
    assert np.shares_memory(rows, m)
