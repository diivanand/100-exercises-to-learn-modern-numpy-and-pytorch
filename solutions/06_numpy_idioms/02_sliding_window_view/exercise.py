# Solution -- 06.02 sliding_window_view

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def windows(x: np.ndarray, width: int) -> np.ndarray:
    """All runs of `width` consecutive values of `x`, as a (n - width + 1, width) VIEW."""
    # No copy: the view's strides overlap, so element [i, j] is x[i + j]
    # read straight from the original buffer. Writing through it is
    # disallowed (the view is read-only) because one source element appears
    # in several windows.
    return sliding_window_view(x, width)


def patches(image: np.ndarray, size: int) -> np.ndarray:
    """Every size x size patch of an image, shape (H-size+1, W-size+1, size, size)."""
    return sliding_window_view(image, (size, size))


def local_means(image: np.ndarray, size: int) -> np.ndarray:
    """The mean of every size x size patch (a "valid" box filter)."""
    # Reduce over the two window axes. The view costs nothing; the mean
    # reads each pixel size*size times, which is fine for small sizes.
    return patches(image, size).mean(axis=(-2, -1))


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
