# Solution -- 01.06 Fancy indexing
import numpy as np


def block(m: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    """The sub-matrix at the given rows and columns, shape (len(rows), len(cols))."""
    # np.ix_ turns two index lists into a (r, 1) and a (1, c) array that
    # broadcast against each other, so every row is paired with every column.
    return m[np.ix_(rows, cols)]


def pick(m: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    """The elements m[rows[i], cols[i]], one per i."""
    return m[rows, cols]


def top_k(xs: np.ndarray, k: int) -> np.ndarray:
    """The `k` largest values of `xs`, largest first."""
    # argsort ascending, take the last k, reverse. (np.argpartition is
    # cheaper for large arrays; 05.01 covers it.)
    order = np.argsort(xs)
    return xs[order[-k:][::-1]]


def test_block_is_a_sub_matrix():
    m = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(block(m, [0, 2], [1, 3]), [[1, 3], [9, 11]])
    assert block(m, [3], [0, 1, 2]).shape == (1, 3)


def test_pick_is_pairs():
    m = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(pick(m, [0, 2], [1, 3]), [1, 11])


def test_fancy_indexing_copies():
    m = np.arange(16).reshape(4, 4)
    b = block(m, [0, 1], [0, 1])
    assert not np.shares_memory(b, m)
    b[:] = -1
    assert m[0, 0] == 0


def test_top_k():
    xs = np.array([3.0, 9.0, 1.0, 7.0, 5.0])
    np.testing.assert_array_equal(top_k(xs, 3), [9.0, 7.0, 5.0])
    np.testing.assert_array_equal(top_k(xs, 1), [9.0])


def test_index_arrays_may_repeat_and_reorder():
    xs = np.array([10, 20, 30])
    np.testing.assert_array_equal(xs[[2, 0, 2, 2]], [30, 10, 30, 30])
