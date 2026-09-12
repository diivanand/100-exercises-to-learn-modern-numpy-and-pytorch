# Solution -- 06.03 meshgrid and grids

from collections.abc import Callable

import numpy as np


def evaluate_on_grid(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray], xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """z[i, j] = f(xs[i], ys[j]) for every pair, as an (len(xs), len(ys)) array."""
    # indexing="ij" makes the first output axis follow the first input, so
    # the result is (len(xs), len(ys)) and z[i, j] means what the docstring
    # says. The default "xy" swaps the first two axes (image convention:
    # rows are y), which is invisible for square grids and wrong otherwise.
    x, y = np.meshgrid(xs, ys, indexing="ij")
    return f(x, y)


def distance_field(height: int, width: int, centre: tuple[float, float]) -> np.ndarray:
    """Euclidean distance of every pixel (row, col) from `centre`."""
    # np.ogrid gives an (h, 1) column and a (1, w) row; broadcasting does
    # the rest without materialising two full h x w coordinate arrays.
    rows, cols = np.ogrid[:height, :width]
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
