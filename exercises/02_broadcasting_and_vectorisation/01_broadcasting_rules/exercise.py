# =============================================================================
#  02.01 -- Broadcasting rules
# =============================================================================
#
#  Broadcasting is how NumPy does arithmetic between arrays of different
#  shapes without copying anything. The rule is short enough to memorise, and
#  worth memorising, because every shape error you will ever meet in NumPy or
#  PyTorch is this rule saying no:
#
#    1. Align the two shapes at their RIGHT-hand end. Pad the shorter one
#       with 1s on the left until the lengths match.
#    2. Walk the axes in pairs. Two lengths are compatible if they are equal,
#       or if either of them is 1. A length of 1 is stretched to the other.
#    3. Anything else is an error.
#
#         (3, 4) + (4,)     ->  (3, 4) + (1, 4)  ->  (3, 4)     fine
#         (3, 4) + (3, 1)   ->                       (3, 4)     fine
#         (3, 4) + (3,)     ->  (3, 4) + (1, 3)  ->  4 vs 3     error
#
#  Johansson, Numerical Python 3rd ed., ch. 2 "Vectorized Expressions" states
#  the rule; McKinney, Python for Data Analysis, ch. 12 "Broadcasting" draws
#  it. The trap in the third line is the one this exercise is about: the
#  per-ROW mean of a (3, 4) matrix has shape (3,), and subtracting it fails.
#  Worse, for a SQUARE matrix it does not fail: (3, 3) - (3,) is legal, the
#  means broadcast along the wrong axis, and every value is quietly wrong.
#  `keepdims=True` keeps the reduced axis as a 1 so the shape is (3, 1) and
#  broadcasting does what you meant.
#
#  TASK
#    Fix `center_rows`, and make `broadcast_shape` implement the rule above
#    so that it agrees with `np.broadcast_shapes`.
#
#  RUN IT
#    ./npt test 02_01
#
# =============================================================================

import numpy as np
import pytest


def center_columns(matrix: np.ndarray) -> np.ndarray:
    """Subtract each column's mean from that column."""
    # (m, n) - (n,) is the case the rule gets right without help.
    return matrix - matrix.mean(axis=0)


def center_rows(matrix: np.ndarray) -> np.ndarray:
    """Subtract each row's mean from that row."""
    # TODO: the means have shape (m,), which aligns with the COLUMNS. Keep
    # the reduced axis so the means are (m, 1).
    return matrix - matrix.mean(axis=1)


def broadcast_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """The shape two arrays would broadcast to, or raise ValueError."""
    # TODO: this aligns the shapes from the LEFT. Pad the shorter shape with
    # 1s on the left instead, so that the trailing axes are compared first.
    result = []
    for i in range(max(len(a_shape), len(b_shape))):
        da = a_shape[i] if i < len(a_shape) else 1
        db = b_shape[i] if i < len(b_shape) else 1
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
