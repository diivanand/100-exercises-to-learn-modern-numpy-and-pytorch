# =============================================================================
#  06.03 -- meshgrid and grids
# =============================================================================
#
#  To evaluate f(x, y) at every combination of xs and ys you need two
#  arrays of the same shape, one holding the x of each point and one the
#  y. `np.meshgrid(xs, ys)` builds them (Johansson, *Numerical Python*
#  ch. 2, "Arrays Filled with Incremental Sequences", and it is used in
#  every plotting example thereafter). What Johansson mentions and most
#  people miss is the `indexing` argument:
#
#      indexing="xy"  (default)  output shape (len(ys), len(xs)): rows are y.
#                                The image and plotting convention.
#      indexing="ij"             output shape (len(xs), len(ys)): the first
#                                axis follows the first input. The matrix
#                                convention, and what z[i, j] = f(xs[i], ys[j])
#                                needs.
#
#  For a square grid the two are transposes of each other and everything
#  looks fine. For a 3 x 2 grid the default produces a 2 x 3 array and
#  z[i, j] indexes out of range or, worse, reads the wrong point.
#
#  Often you do not need the full coordinate arrays at all. `np.ogrid[:h,
#  :w]` returns an (h, 1) column and a (1, w) row; any arithmetic between
#  them broadcasts (02.01) to (h, w) on the fly. A distance field over a
#  4000 x 4000 image needs no 128 MB of coordinates first. (np.mgrid is the
#  same thing materialised, and np.indices is its cousin.)
#
#  TASK
#    Make `evaluate_on_grid` honour its docstring for non-square grids, and
#    write `distance_field` with np.ogrid.
#
#  RUN IT
#    ./npt test 06_03
#
# =============================================================================

from collections.abc import Callable

import numpy as np


def evaluate_on_grid(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray], xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """z[i, j] = f(xs[i], ys[j]) for every pair, as an (len(xs), len(ys)) array."""
    # TODO: default "xy" indexing: the result is (len(ys), len(xs)).
    x, y = np.meshgrid(xs, ys)
    return f(x, y)


def distance_field(height: int, width: int, centre: tuple[float, float]) -> np.ndarray:
    """Euclidean distance of every pixel (row, col) from `centre`."""
    # TODO: np.ogrid, and let broadcasting build the (height, width) result.
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.sqrt((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)


def test_evaluate_on_grid_indexing():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0])
    z = evaluate_on_grid(lambda x, y: x + y, xs, ys)
    assert z.shape == (3, 2), "z[i, j] pairs xs[i] with ys[j]"
    np.testing.assert_array_equal(z, [[10, 20], [11, 21], [12, 22]])


def test_evaluate_on_grid_matches_a_loop():
    xs = np.linspace(-1.0, 1.0, 7)
    ys = np.linspace(0.0, 2.0, 4)
    z = evaluate_on_grid(lambda x, y: np.sin(x) * y**2, xs, ys)
    expected = np.array([[np.sin(x) * y**2 for y in ys] for x in xs])
    np.testing.assert_allclose(z, expected)


def test_distance_field():
    d = distance_field(3, 4, (1.0, 1.0))
    assert d.shape == (3, 4)
    assert d[1, 1] == 0.0
    np.testing.assert_allclose(d[0, 0], np.sqrt(2.0))
    np.testing.assert_allclose(d[2, 3], np.sqrt(1.0 + 4.0))


def test_distance_field_does_not_build_full_coordinate_arrays(monkeypatch):
    # np.ogrid (open grid) is the point: it returns broadcastable (h, 1) and
    # (1, w) arrays. meshgrid would allocate two full h x w arrays first.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("use np.ogrid, not np.meshgrid, for a distance field")

    monkeypatch.setattr(np, "meshgrid", forbidden)
    assert distance_field(5, 6, (2.0, 2.0)).shape == (5, 6)
