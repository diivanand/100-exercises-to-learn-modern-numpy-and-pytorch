# Solution -- 04.05 default_rng

import numpy as np


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` points in the plane and an integer label in {0, 1, 2} for each."""
    # A Generator is a local object: two functions with different seeds do
    # not interfere, and neither touches the global legacy state.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = rng.integers(0, 3, size=n)  # high is exclusive, like range()
    return x, y


def roll_dice(n: int, seed: int) -> np.ndarray:
    """`n` fair six-sided dice rolls, values 1 to 6 inclusive."""
    rng = np.random.default_rng(seed)
    # `endpoint=True` is the honest way to say "6 is a possible outcome";
    # the alternative `integers(1, 7)` works but reads as an off-by-one.
    return rng.integers(1, 6, size=n, endpoint=True)


def bootstrap_means(x: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """The mean of `resamples` resamples-with-replacement of `x`."""
    rng = np.random.default_rng(seed)
    # One call draws a (resamples, n) matrix of indices; no Python loop.
    samples = rng.choice(x, size=(resamples, x.size), replace=True)
    return samples.mean(axis=1)


def shuffled_copy(x: np.ndarray, seed: int) -> np.ndarray:
    """A shuffled copy of `x`. The caller's array is left alone."""
    rng = np.random.default_rng(seed)
    # permutation returns a new array; shuffle would reorder `x` in place.
    return rng.permutation(x)


def test_make_dataset_is_reproducible_and_seed_dependent():
    x1, y1 = make_dataset(100, seed=7)
    x2, y2 = make_dataset(100, seed=7)
    x3, _ = make_dataset(100, seed=8)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(x1, x3)
    assert x1.shape == (100, 2) and y1.shape == (100,)
    assert set(np.unique(y1)) == {0, 1, 2}


def test_dice_reach_six():
    rolls = roll_dice(2000, seed=0)
    assert rolls.min() == 1
    assert rolls.max() == 6, "6 must be a possible outcome"
    assert rolls.dtype.kind == "i"


def test_bootstrap_means_have_the_right_spread():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(400)
    means = bootstrap_means(x, resamples=2000, seed=1)
    assert means.shape == (2000,)
    # Standard error of the mean of 400 unit-variance samples is 1/20.
    assert abs(means.mean() - x.mean()) < 0.01
    assert abs(means.std() - 0.05) < 0.01


def test_shuffled_copy_leaves_the_input_alone():
    x = np.arange(20)
    before = x.copy()
    y = shuffled_copy(x, seed=3)
    np.testing.assert_array_equal(x, before)
    assert sorted(y) == list(range(20))
    assert not np.array_equal(x, y)


def test_global_legacy_state_is_not_touched():
    # The legacy API (np.random.seed, np.random.rand, ...) is one hidden
    # global generator. Code that uses it makes every other user of that
    # generator -- a library, a test, a colleague's notebook cell -- depend on
    # what you did last. A Generator object has no such reach.
    before = np.random.get_bit_generator().state["state"]["key"].copy()
    make_dataset(10, seed=1)
    roll_dice(10, seed=2)
    bootstrap_means(np.arange(5.0), 3, seed=4)
    shuffled_copy(np.arange(5), seed=5)
    after = np.random.get_bit_generator().state["state"]["key"]
    np.testing.assert_array_equal(before, after)
