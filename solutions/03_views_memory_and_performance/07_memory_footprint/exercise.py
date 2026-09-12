# Solution -- 03.07 Memory footprint
import math

import numpy as np


def footprint_bytes(shape: tuple[int, ...], dtype: np.dtype | type) -> int:
    """How many bytes an array of this shape and dtype occupies."""
    # np.dtype(...) accepts anything NumPy accepts as a dtype: np.float32,
    # "int16", np.dtype("f8"). itemsize is the width of one element.
    return math.prod(shape) * np.dtype(dtype).itemsize


def make_grid(n: int) -> np.ndarray:
    """An (n, n) grid of x-coordinates in [0, 1), in single precision."""
    # A coordinate grid for plotting or a kernel does not need 16 digits.
    # float32 halves the memory and the bandwidth, and on a GPU it is the
    # native speed (17.03). Ask for the dtype at creation: converting after
    # the fact means the float64 array existed anyway.
    row = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    return np.broadcast_to(row, (n, n)).copy()


def smallest_uint(max_value: int) -> type:
    """The narrowest unsigned integer dtype that can hold 0..max_value."""
    # np.iinfo knows each integer dtype's range; ask rather than remember.
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(candidate).max:
            return candidate
    raise ValueError(f"{max_value} does not fit in any unsigned integer dtype")


def test_footprint_matches_nbytes():
    for shape, dtype in [((3, 4), np.float64), ((10,), "int16"), ((2, 2, 2), np.bool_)]:
        assert footprint_bytes(shape, dtype) == np.empty(shape, dtype=dtype).nbytes


def test_footprint_of_the_big_grid():
    # 20 000 x 20 000 doubles is 3.2 GB. Singles: 1.6 GB. Computed, not built.
    assert footprint_bytes((20_000, 20_000), np.float64) == 3_200_000_000
    assert footprint_bytes((20_000, 20_000), np.float32) == 1_600_000_000


def test_make_grid_is_single_precision():
    grid = make_grid(256)
    assert grid.dtype == np.float32
    assert grid.nbytes == 256 * 256 * 4
    assert grid.shape == (256, 256)


def test_make_grid_values():
    grid = make_grid(4)
    np.testing.assert_allclose(grid[0], [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(grid[3], grid[0])


def test_smallest_uint():
    assert smallest_uint(255) is np.uint8
    assert smallest_uint(256) is np.uint16
    assert smallest_uint(70_000) is np.uint32
    assert smallest_uint(2**40) is np.uint64


def test_the_ranges_are_what_iinfo_says():
    # Not a task. Fixed-width integers wrap; know the edges (01.02).
    assert np.iinfo(np.uint8).max == 255
    assert np.iinfo(np.int8).min == -128
    assert np.finfo(np.float32).eps > np.finfo(np.float64).eps
