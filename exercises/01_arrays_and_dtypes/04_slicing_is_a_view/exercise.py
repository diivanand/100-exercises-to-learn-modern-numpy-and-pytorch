# =============================================================================
#  01.04 -- Slicing is a view
# =============================================================================
#
#  Slicing a Python list copies. Slicing a NumPy array does not: `xs[3:7]` is
#  a new ndarray object that points INTO the memory of `xs`, with its own
#  offset, shape and strides. Johansson, ch. 2, "Views": "Operations that
#  only require changing the strides attribute result in new ndarray objects
#  that refer to the same data as the original array. Such arrays are called
#  views."
#
#  This is a feature. It is why `m[:, 0]` costs nothing however big `m` is,
#  why `xs[::-1]` does not copy, and why a function can hand you a row of a
#  gigabyte matrix without a gigabyte of traffic. It is also the single most
#  common source of "my data changed and I do not know why" in NumPy code.
#
#  The rules:
#
#   * Basic indexing -- integers, slices, `...`, `None` -- gives a view.
#   * Arithmetic (`xs + 1`) and integer- or boolean-array indexing (01.05,
#     01.06) give a copy.
#   * `.copy()` gives a copy. If a function's contract is "you may modify the
#     result", that is the line that makes it true.
#   * `np.shares_memory(a, b)` answers the question when you are not sure.
#     Use it in tests.
#
#  The mirror-image mistake is assigning to a NAME instead of to the data.
#  Inside a function, `xs = xs[::-1]` rebinds the local variable to a new
#  view; the caller's array is exactly as it was. Writing INTO an array is
#  spelt `xs[:] = ...` (or `xs[...] = ...`). Effective Python, 3rd ed., Item
#  30 ("Know That Function Arguments Can Be Mutated") is the Python-level
#  version of this distinction.
#
#  TASK
#    Make `window` return something safe to modify, and make
#    `reverse_in_place` actually reverse the caller's array. Then write
#    `every_other_row` as a view, not a copy.
#
#  RUN IT
#    ./npt test 01_04
#
# =============================================================================

import numpy as np


def window(xs: np.ndarray, start: int, stop: int) -> np.ndarray:
    """An independent copy of `xs[start:stop]` the caller may modify freely."""
    # TODO: this is a view. Writing into it writes into `xs`.
    return xs[start:stop]


def reverse_in_place(xs: np.ndarray) -> None:
    """Reverse `xs` in place, so the caller's array is reversed."""
    # TODO: this rebinds the local name `xs` to a new view. The caller's
    # array does not change. Assign into the array, not to the name.
    xs = xs[::-1]


def every_other_row(m: np.ndarray) -> np.ndarray:
    """Rows 0, 2, 4, ... of `m`, as a view (no copy)."""
    # TODO: a slice with a step.
    raise NotImplementedError


def test_basic_slices_are_views():
    xs = np.arange(10)
    middle = xs[3:7]
    assert np.shares_memory(middle, xs)
    middle[0] = 99
    assert xs[3] == 99
    # Even a reversed or strided slice is a view: only the strides change.
    assert np.shares_memory(xs[::-1], xs)
    assert np.shares_memory(xs[::3], xs)


def test_window_is_safe_to_modify():
    xs = np.arange(10.0)
    w = window(xs, 2, 5)
    np.testing.assert_array_equal(w, [2.0, 3.0, 4.0])
    w[:] = -1.0
    np.testing.assert_array_equal(xs, np.arange(10.0))


def test_reverse_in_place_changes_the_callers_array():
    xs = np.array([1, 2, 3, 4])
    result = reverse_in_place(xs)
    assert result is None
    np.testing.assert_array_equal(xs, [4, 3, 2, 1])


def test_every_other_row_is_a_view():
    m = np.arange(20).reshape(5, 4)
    rows = every_other_row(m)
    assert rows.shape == (3, 4)
    np.testing.assert_array_equal(rows[1], [8, 9, 10, 11])
    assert np.shares_memory(rows, m)
