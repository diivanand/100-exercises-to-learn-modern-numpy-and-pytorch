# Solution -- 06.01 cumsum and diff

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of each run of `window` consecutive values: len(x) - window + 1 results."""
    # With a leading zero, c[i] is the sum of the first i elements, so the
    # sum of x[i:i+w] is c[i+w] - c[i] for every i in one vectorised step.
    # Without the zero, the first window is lost and the count is off by one.
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[window:] - c[:-window]) / window


def changes(x: np.ndarray) -> np.ndarray:
    """x[i] - x[i-1] for every i, with the first change taken from 0."""
    # `prepend` supplies the value before x[0], so the output has the same
    # length as x -- a change for every sample, including the first.
    return np.diff(x, prepend=0.0)


def reconstruct(changes_: np.ndarray) -> np.ndarray:
    """Undo `changes`: the series whose changes-from-zero are `changes_`."""
    return np.cumsum(changes_)


def test_moving_average_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    for w in (1, 3, 7):
        expected = np.array([x[i : i + w].mean() for i in range(len(x) - w + 1)])
        np.testing.assert_allclose(moving_average(x, w), expected)


def test_moving_average_includes_the_first_window():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(moving_average(x, 2), [1.5, 2.5, 3.5, 4.5])
    assert moving_average(x, 5).shape == (1,)
    np.testing.assert_allclose(moving_average(x, 5), [3.0])


def test_changes_and_reconstruct_are_inverses():
    rng = np.random.default_rng(1)
    x = rng.integers(-5, 5, size=50).astype(float)
    d = changes(x)
    assert d.shape == x.shape
    assert d[0] == x[0]
    np.testing.assert_allclose(reconstruct(d), x)
