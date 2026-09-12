# Solution -- 03.02 Strides
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def c_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """The strides, in bytes, of a C-ordered array of this shape."""
    # The last axis moves one item at a time; each earlier axis moves by the
    # whole extent of everything after it. Build from the right.
    strides = []
    step = itemsize
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return tuple(reversed(strides))


def sliding_windows(xs: np.ndarray, width: int) -> np.ndarray:
    """All contiguous windows of `width` elements, as a (n - width + 1, width) view."""
    # sliding_window_view is as_strided with the arithmetic done for you and
    # the result marked read-only. Row i is xs[i : i + width]; no element is
    # copied, so the (n, width) result costs nothing however large width is.
    return sliding_window_view(xs, width)


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
