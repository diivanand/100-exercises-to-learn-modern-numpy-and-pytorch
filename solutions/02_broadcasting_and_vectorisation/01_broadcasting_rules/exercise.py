# Solution -- 02.01 Broadcasting rules
import numpy as np
import pytest


def center_columns(matrix: np.ndarray) -> np.ndarray:
    """Subtract each column's mean from that column."""
    # (m, n) - (n,): the (n,) is padded on the LEFT to (1, n), which stretches
    # down the rows. That is the case broadcasting gets right without help.
    return matrix - matrix.mean(axis=0)


def center_rows(matrix: np.ndarray) -> np.ndarray:
    """Subtract each row's mean from that row."""
    # (m, n) - (m,) would pad to (1, m): wrong axis, and for a square matrix
    # it does not even fail. keepdims keeps the reduced axis as length 1, so
    # the means are (m, 1) and stretch across the columns as intended.
    return matrix - matrix.mean(axis=1, keepdims=True)


def broadcast_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """The shape two arrays would broadcast to, or raise ValueError."""
    # Right-align the shapes by padding the shorter with 1s on the left, then
    # walk the axes: equal is fine, 1 stretches, anything else is an error.
    ndim = max(len(a_shape), len(b_shape))
    a = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
    result = []
    for da, db in zip(a, b, strict=True):
        if da == db or db == 1:
            result.append(da)
        elif da == 1:
            result.append(db)
        else:
            raise ValueError(f"shapes {a_shape} and {b_shape} are not broadcastable")
    return tuple(result)


def test_center_columns():
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    np.testing.assert_allclose(center_columns(m), [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(center_columns(m).mean(axis=0), 0.0, atol=1e-12)


def test_center_rows_on_a_rectangular_matrix():
    m = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    np.testing.assert_allclose(center_rows(m), [[-1.0, 0.0, 1.0], [-10.0, 0.0, 10.0]])


def test_center_rows_on_a_square_matrix():
    # The shortcut `matrix - matrix.mean(axis=1)` does not raise here: the
    # (3,) means broadcast along the wrong axis and every value is wrong.
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    centred = center_rows(m)
    np.testing.assert_allclose(centred.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(centred[0], [-1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((3, 4), (4,)),
        ((3, 4), (3, 1)),
        ((8, 1, 6, 1), (7, 1, 5)),
        ((5,), (1,)),
        ((2, 1), (1, 3)),
        ((), (3, 2)),
    ],
)
def test_broadcast_shape_agrees_with_numpy(a, b):
    assert broadcast_shape(a, b) == np.broadcast_shapes(a, b)


def test_broadcast_shape_aligns_from_the_right():
    # (3,) against (3, 4) is an error, not (3, 4): the trailing axes are
    # compared first. Aligning from the left gets this one silently wrong.
    with pytest.raises(ValueError):
        broadcast_shape((3,), (3, 4))
    with pytest.raises(ValueError):
        np.broadcast_shapes((3,), (3, 4))
    assert broadcast_shape((4,), (3, 4)) == (3, 4)


def test_broadcast_shape_rejects_mismatches():
    with pytest.raises(ValueError):
        broadcast_shape((2, 3), (3, 2))
