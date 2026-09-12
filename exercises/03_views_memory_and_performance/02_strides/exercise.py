# =============================================================================
#  03.02 -- Strides
# =============================================================================
#
#  `a.strides` is a tuple with one entry per axis: how many BYTES to step
#  in memory to move one position along that axis. A C-ordered (2, 3) array
#  of int32 has strides (12, 4): the next column is 4 bytes on, the next row
#  is 3 * 4 = 12 bytes on. Johansson, Numerical Python 3rd ed., ch. 2 "Order
#  of Array Data in Memory" works this example and then makes the point
#  that matters: many operations are IMPLEMENTED as a change of strides.
#  `a.T` swaps them. `a[::2]` doubles one. `a[:, None]` inserts a 0. None of
#  these touch the data, which is why they are free and why they are views
#  (03.01). McKinney, Python for Data Analysis, ch. 12 "ndarray Object
#  Internals" draws the same picture.
#
#  A stride may even be ZERO: then moving along that axis stays put, and
#  the same element is read again. That is how broadcasting works without
#  copying (`np.broadcast_to` gives you such an array), and it is how
#  sliding windows work: a (n - w + 1, w) array whose two strides are BOTH
#  the itemsize reads xs[i + j] at position (i, j). Every window overlaps
#  the next, and not one element is copied.
#
#  You can build that by hand with `np.lib.stride_tricks.as_strided`, and
#  people do, and then they count strides in elements instead of bytes and
#  read memory the array does not own. as_strided does no checking at all.
#  The modern spelling is `np.lib.stride_tricks.sliding_window_view(xs,
#  width)`: same result, arithmetic done for you, and the view is marked
#  read-only, because a write through one window would silently change
#  every window that shares the element.
#
#  TASK
#    Make `c_strides` compute the strides of a C-ordered array, and
#    rewrite `sliding_windows` with sliding_window_view.
#
#  RUN IT
#    ./npt test 03_02
#
# =============================================================================

import numpy as np
from numpy.lib.stride_tricks import as_strided


def c_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """The strides, in bytes, of a C-ordered array of this shape."""
    # TODO: the last axis steps one item; each earlier axis steps by the
    # product of everything after it. Work from the right.
    return tuple(itemsize * extent for extent in shape)


def sliding_windows(xs: np.ndarray, width: int) -> np.ndarray:
    """All contiguous windows of `width` elements, as a (n - width + 1, width) view."""
    # TODO: strides are in BYTES, not elements, so (1, 1) is only right for
    # one-byte dtypes -- with float64 this reads misaligned garbage. And the
    # result is writeable, which it must not be. Use sliding_window_view.
    n = len(xs)
    return as_strided(xs, shape=(n - width + 1, width), strides=(1, 1))


def test_c_strides_match_numpy():
    for shape in [(5,), (2, 3), (4, 3, 2), (1, 7, 1)]:
        for dtype in [np.int8, np.float32, np.float64]:
            a = np.empty(shape, dtype=dtype)
            assert c_strides(shape, a.itemsize) == a.strides, (shape, dtype)


def test_transpose_is_free():
    # Not a task. Swapping the strides is all a transpose does.
    m = np.arange(6.0).reshape(2, 3)
    assert m.T.strides == m.strides[::-1]
    assert np.shares_memory(m.T, m)


def test_sliding_windows_values():
    xs = np.arange(6)
    windows = sliding_windows(xs, 3)
    np.testing.assert_array_equal(windows, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def test_sliding_windows_respect_the_itemsize():
    # Strides are in BYTES. Counting in elements is right for int8 and wrong
    # for everything else, which is why this test uses float64.
    xs = np.linspace(0.0, 1.0, 8)
    windows = sliding_windows(xs, 4)
    expected = np.array([xs[i : i + 4] for i in range(5)])
    np.testing.assert_array_equal(windows, expected)


def test_sliding_windows_do_not_copy():
    xs = np.arange(1000.0)
    windows = sliding_windows(xs, 100)
    assert windows.shape == (901, 100)
    assert np.shares_memory(windows, xs)
    assert windows.strides == (xs.itemsize, xs.itemsize)


def test_sliding_windows_are_read_only():
    # Every element appears in several windows, so a write through one would
    # silently change others. The view must refuse.
    windows = sliding_windows(np.arange(5.0), 2)
    assert not windows.flags.writeable
