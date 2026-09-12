# =============================================================================
#  01.01 -- Creating arrays
# =============================================================================
#
#  A NumPy array is a block of memory plus metadata: `shape`, `dtype`,
#  `strides` (Johansson, Numerical Python 3rd ed., ch. 2, "The NumPy Array
#  Object", Table 2-1). Every element has the same type, which is what makes
#  the block contiguous, and contiguity is what makes everything fast.
#
#  You will build arrays from Python lists only for examples and tests. Real
#  arrays come from files, from other arrays, or from the generators in
#  Johansson's Table 2-3:
#
#      np.zeros(shape), np.ones(shape), np.full(shape, value), np.empty(shape)
#      np.arange(start, stop, step)     like range: stop is EXCLUDED
#      np.linspace(start, stop, n)      n points: stop is INCLUDED
#      np.eye(n), np.identity(n), np.diag(v)
#
#  Two of these bite:
#
#   * `np.arange` with a float step computes the length as
#     ceil((stop - start) / step). With step = 0.01 that division is
#     101.00000000000001, so you get 102 points and a last element past the
#     stop. When you want "n points from a to b", say so: `np.linspace`.
#     (Johansson, ch. 2, "Arrays Filled with Incremental Sequences".)
#
#   * `np.empty` does not zero its memory. It is faster for exactly that
#     reason, and it is only correct when every element is written before it
#     is read. An accumulator is not that case.
#
#  The dtype is part of what you are creating. `np.zeros(shape)` is float64;
#  `np.full(shape, 200)` is int64 because that is what a Python int becomes.
#  An image that is going to be handed to something expecting bytes needs
#  `dtype=np.uint8` said out loud (01.02 is about what happens when you get
#  this wrong).
#
#  TASK
#    Fix `unit_grid` and `accumulator`. Then write `constant_image` and
#    `one_hot` using `np.full` and `np.eye`.
#
#  RUN IT
#    ./npt test 01_01
#
# =============================================================================

import numpy as np


def unit_grid(n: int) -> np.ndarray:
    """`n` evenly spaced points from 0.0 to 1.0 inclusive."""
    # TODO: arange with a float step. Its length is computed from a division
    # that is not exact, so for some n this has n + 1 points.
    step = 1.0 / (n - 1)
    return np.arange(0.0, 1.0 + step, step)


def accumulator(shape: tuple[int, ...]) -> np.ndarray:
    """A zeroed float64 array of the given shape, ready to be added into."""
    # TODO: this is an integer array. Adding 0.5 into it raises, because
    # NumPy will not silently cast float to int under `+=`.
    return np.zeros(shape, dtype=int)


def constant_image(height: int, width: int, value: int) -> np.ndarray:
    """A `height` x `width` uint8 image filled with `value`."""
    # TODO: use np.full, and say the dtype.
    raise NotImplementedError


def one_hot(index: int, n: int) -> np.ndarray:
    """Row `index` of the `n` x `n` identity: a vector with a single 1.0."""
    # TODO: use np.eye.
    raise NotImplementedError


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
