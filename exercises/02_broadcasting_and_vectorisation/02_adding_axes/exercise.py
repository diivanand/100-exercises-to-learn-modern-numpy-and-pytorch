# =============================================================================
#  02.02 -- Adding axes
# =============================================================================
#
#  Broadcasting pads shapes on the left (02.01). When you need a 1 somewhere
#  ELSE, you put it there yourself, and there are three spellings:
#
#      v[:, None]                 (n,)  ->  (n, 1)     a column
#      v[None, :]                 (n,)  ->  (1, n)     a row
#      np.expand_dims(v, axis=1)  (n,)  ->  (n, 1)     the same, by name
#
#  `None` in an index means "insert an axis of length 1 here"; `np.newaxis`
#  is an alias for None that some people find more readable. Johansson,
#  Numerical Python 3rd ed., ch. 2 "Reshaping and Resizing" walks through
#  both spellings. All of them are views: the data is not touched, only the
#  shape and strides (03.02).
#
#  Why it matters: a per-row scale factor has shape (m,), and multiplying a
#  (m, n) matrix by it aligns it with the COLUMNS. To scale rows you need it
#  as (m, 1). And a column plus a row, (n, 1) + (1, m), broadcasts to the
#  whole (n, m) table of pairwise sums -- the trick behind every pairwise
#  computation, which 02.06 builds on.
#
#  The third function is a shape convention rather than a broadcasting
#  problem: image models expect a stack of images as (N, C, H, W), batch
#  first, channels SECOND. Godoy, Deep Learning with PyTorch Step-by-Step,
#  ch. 4 "Shape (NCHW vs NHWC)" explains why it trips everyone up once.
#
#  TASK
#    Fix the three functions.
#
#  RUN IT
#    ./npt test 02_02
#
# =============================================================================

import numpy as np


def scale_rows(matrix: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Multiply row i of `matrix` by `scale[i]`."""
    # TODO: `scale` is (m,), which lines up with the columns. Make it (m, 1).
    return matrix * scale


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The (len(a), len(b)) table of a[i] + b[j]."""
    # TODO: this is an elementwise sum, which needs len(a) == len(b) and
    # gives a 1-D result. A column plus a row gives the table.
    return a + b


def add_channel_axis(images: np.ndarray) -> np.ndarray:
    """Turn a stack of greyscale images (N, H, W) into (N, 1, H, W)."""
    # TODO: this appends the axis at the END, giving (N, H, W, 1): NHWC.
    # The channel axis belongs after the batch axis.
    return images[..., None]


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
