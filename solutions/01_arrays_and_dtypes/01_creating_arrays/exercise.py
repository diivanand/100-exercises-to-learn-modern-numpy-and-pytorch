# Solution -- 01.01 Creating arrays
import numpy as np


def unit_grid(n: int) -> np.ndarray:
    """`n` evenly spaced points from 0.0 to 1.0 inclusive."""
    # linspace takes a COUNT and includes the endpoint; the last element is
    # exactly 1.0 because linspace computes it as start + (n-1)*step and then
    # overwrites the final entry with `stop` outright.
    return np.linspace(0.0, 1.0, n)


def accumulator(shape: tuple[int, ...]) -> np.ndarray:
    """A zeroed float64 array of the given shape, ready to be added into."""
    return np.zeros(shape, dtype=np.float64)


def constant_image(height: int, width: int, value: int) -> np.ndarray:
    """A `height` x `width` uint8 image filled with `value`."""
    # np.full takes the fill value; the dtype is what the caller's data is,
    # not what NumPy would infer from a Python int (int64).
    return np.full((height, width), value, dtype=np.uint8)


def one_hot(index: int, n: int) -> np.ndarray:
    """Row `index` of the `n` x `n` identity: a vector with a single 1.0."""
    return np.eye(n)[index]


def test_unit_grid_has_the_requested_count_and_endpoint():
    grid = unit_grid(11)
    assert grid.shape == (11,)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    np.testing.assert_allclose(np.diff(grid), 0.1)


def test_unit_grid_is_not_fooled_by_floating_point():
    # 1/100 is not representable in binary. Anything that computes the
    # number of points from `stop / step` can get 102 here.
    grid = unit_grid(101)
    assert grid.shape == (101,)
    assert grid[-1] == 1.0
    assert unit_grid(2).tolist() == [0.0, 1.0]


def test_accumulator_is_zero_and_float():
    acc = accumulator((2, 3))
    assert acc.dtype == np.float64
    np.testing.assert_array_equal(acc, np.zeros((2, 3)))
    acc += 0.5  # would raise for an integer array: cannot cast float to int
    np.testing.assert_array_equal(acc, np.full((2, 3), 0.5))


def test_constant_image_dtype_and_value():
    image = constant_image(2, 4, 200)
    assert image.shape == (2, 4)
    assert image.dtype == np.uint8
    assert image.nbytes == 8
    assert (image == 200).all()


def test_one_hot():
    np.testing.assert_array_equal(one_hot(2, 4), [0.0, 0.0, 1.0, 0.0])
    assert one_hot(0, 3).dtype == np.float64
