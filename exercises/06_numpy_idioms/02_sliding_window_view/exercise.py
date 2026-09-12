# =============================================================================
#  06.02 -- sliding_window_view
# =============================================================================
#
#  An array is a pointer, a shape and STRIDES: the number of bytes to step
#  along each axis (03.02). Nothing says the strides have to be disjoint. A
#  2-D view of a 1-D array with strides (8, 8) has rows that overlap by all
#  but one element, which is exactly what "every window of width w" means,
#  and it costs no memory at all. McKinney, *Python for Data Analysis*
#  ch. 12, shows this with `np.lib.stride_tricks.as_strided`, a function
#  that trusts you completely and reads garbage if you miscount.
#
#  `np.lib.stride_tricks.sliding_window_view(x, shape)` (NumPy 1.20) does
#  the counting for you: pass the window shape, get a read-only view whose
#  last axes are the window. For a 1-D signal that is (n - w + 1, w); for a
#  2-D image with window (k, k) it is (H - k + 1, W - k + 1, k, k), every
#  patch, no copy. Reduce over the window axes and you have a box filter;
#  a moving average is the 1-D case (06.01 does it with cumsum, which is
#  cheaper for large windows; this is simpler and general).
#
#  The tests check `np.shares_memory`, because the point is that no copy
#  is made: a list comprehension over slices gives the same numbers and a
#  thousand times the memory.
#
#  TASK
#    Build `windows` and `patches` as views, and `local_means` on top.
#
#  RUN IT
#    ./npt test 06_02
#
# =============================================================================

import numpy as np


def windows(x: np.ndarray, width: int) -> np.ndarray:
    """All runs of `width` consecutive values of `x`, as a (n - width + 1, width) VIEW."""
    # TODO: this copies every window into a new array. Use
    # np.lib.stride_tricks.sliding_window_view.
    return np.array([x[i : i + width] for i in range(len(x) - width + 1)])


def patches(image: np.ndarray, size: int) -> np.ndarray:
    """Every size x size patch of an image, shape (H-size+1, W-size+1, size, size)."""
    # TODO: sliding_window_view with a 2-D window shape.
    h, w = image.shape
    return np.array(
        [
            [image[i : i + size, j : j + size] for j in range(w - size + 1)]
            for i in range(h - size + 1)
        ]
    )


def local_means(image: np.ndarray, size: int) -> np.ndarray:
    """The mean of every size x size patch (a "valid" box filter)."""
    # TODO: reduce the patches over their two window axes.
    return image


def test_windows_values_and_shape():
    x = np.arange(6.0)
    w = windows(x, 3)
    assert w.shape == (4, 3)
    np.testing.assert_array_equal(w, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def test_windows_is_a_view_not_a_copy():
    x = np.arange(1000.0)
    w = windows(x, 10)
    assert np.shares_memory(x, w), "build the windows with sliding_window_view"
    # Overlapping windows: consecutive rows are one element apart in memory.
    assert w.strides == (x.strides[0], x.strides[0])
    x[5] = -1.0
    assert w[5, 0] == -1.0 and w[0, 5] == -1.0


def test_patches_and_local_means():
    image = np.arange(20.0).reshape(4, 5)
    p = patches(image, 2)
    assert p.shape == (3, 4, 2, 2)
    np.testing.assert_array_equal(p[1, 2], image[1:3, 2:4])
    assert np.shares_memory(image, p)
    means = local_means(image, 2)
    expected = np.array(
        [[image[i : i + 2, j : j + 2].mean() for j in range(4)] for i in range(3)]
    )
    np.testing.assert_allclose(means, expected)
