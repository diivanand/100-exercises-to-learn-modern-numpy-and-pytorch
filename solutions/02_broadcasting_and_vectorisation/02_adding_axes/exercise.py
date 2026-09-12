# Solution -- 02.02 Adding axes
import numpy as np


def scale_rows(matrix: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Multiply row i of `matrix` by `scale[i]`."""
    # `scale` is (m,). Written as (m, 1) it lines up with the rows; written
    # bare it would line up with the columns, which is a different function.
    return matrix * scale[:, None]


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The (len(a), len(b)) table of a[i] + b[j]."""
    # A column (n, 1) plus a row (1, m) broadcasts to (n, m). This is the
    # whole trick behind pairwise tables, and 02.06 uses it for distances.
    return a[:, None] + b[None, :]


def add_channel_axis(images: np.ndarray) -> np.ndarray:
    """Turn a stack of greyscale images (N, H, W) into (N, 1, H, W)."""
    # The channel axis goes SECOND: PyTorch and most image code expect NCHW
    # (batch, channel, height, width). np.expand_dims names the position;
    # `images[:, None]` is the same thing spelled with indexing. Both are
    # views, so no pixels are copied.
    return np.expand_dims(images, axis=1)


def test_scale_rows():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        scale_rows(m, np.array([1.0, 10.0])), [[1, 2, 3], [40, 50, 60]]
    )


def test_scale_rows_on_a_square_matrix():
    # Square is where the wrong axis does not raise, so it is what we test.
    m = np.ones((3, 3))
    np.testing.assert_allclose(
        scale_rows(m, np.array([1.0, 2.0, 3.0]))[:, 0], [1.0, 2.0, 3.0]
    )


def test_outer_sum():
    table = outer_sum(np.array([0, 10, 20]), np.array([1, 2]))
    np.testing.assert_array_equal(table, [[1, 2], [11, 12], [21, 22]])


def test_outer_sum_with_different_lengths():
    table = outer_sum(np.arange(5), np.arange(3))
    assert table.shape == (5, 3)
    assert table[4, 2] == 6


def test_add_channel_axis_is_nchw():
    images = np.zeros((8, 28, 28))
    out = add_channel_axis(images)
    assert out.shape == (8, 1, 28, 28)


def test_add_channel_axis_does_not_copy():
    images = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = add_channel_axis(images)
    assert np.shares_memory(out, images)
    out[1, 0, 2, 3] = -1.0
    assert images[1, 2, 3] == -1.0
