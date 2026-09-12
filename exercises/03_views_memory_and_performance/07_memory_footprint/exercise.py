# =============================================================================
#  03.07 -- Memory footprint
# =============================================================================
#
#  An array's memory is its element count times its `itemsize`, and
#  `a.nbytes` reports it. That is the whole formula, and it is worth doing
#  in your head before creating anything large: a 20 000 x 20 000 grid of
#  float64 is 3.2 GB, and the expression that builds it will make a
#  temporary or two of the same size on the way (03.05).
#
#  The dtype is the lever. Johansson, Numerical Python 3rd ed., ch. 2 "Data
#  Types" lists them; the ones that matter are float64 (8 bytes, 16
#  significant digits), float32 (4 bytes, 7 digits), and the integers from
#  int8 to int64. NumPy defaults to float64 and int64, which is the safe
#  choice and often twice or eight times more than the data deserves: pixel
#  values fit in uint8, a coordinate grid for plotting is fine in float32,
#  and on a GPU float32 is the native speed while float64 is up to sixty
#  four times slower on consumer cards (the C++ course's 17.03). Ask for the
#  dtype AT CREATION, with the `dtype=` argument every constructor takes; a
#  cast afterwards means the big version existed anyway.
#
#  Choosing an integer width means knowing the range, and the way to know
#  it is to ask: `np.iinfo(np.uint16).max` is 65535, `np.finfo(np.float32)`
#  has `.eps`, `.max` and `.tiny`. 01.02 showed what happens when a value
#  does not fit.
#
#  TASK
#    Make `make_grid` single precision, and make `smallest_uint` choose
#    the narrowest type that fits.
#
#  RUN IT
#    ./npt test 03_07
#
# =============================================================================

import math

import numpy as np


def footprint_bytes(shape: tuple[int, ...], dtype: np.dtype | type) -> int:
    """How many bytes an array of this shape and dtype occupies."""
    return math.prod(shape) * np.dtype(dtype).itemsize


def make_grid(n: int) -> np.ndarray:
    """An (n, n) grid of x-coordinates in [0, 1), in single precision."""
    # TODO: this builds the grid in float64 and never asks for anything
    # else. Pass dtype=np.float32 where the values are created.
    row = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.broadcast_to(row, (n, n)).copy()


def smallest_uint(max_value: int) -> type:
    """The narrowest unsigned integer dtype that can hold 0..max_value."""
    # TODO: always answering uint64 is safe and wasteful. Try the types from
    # narrowest to widest and return the first whose np.iinfo(...).max is
    # large enough; raise ValueError if none is.
    return np.uint64


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
