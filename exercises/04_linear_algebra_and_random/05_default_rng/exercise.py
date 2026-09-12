# =============================================================================
#  04.05 -- default_rng
# =============================================================================
#
#  Every NumPy tutorial written before 2019 seeds one hidden global generator
#  and draws from it:
#
#      np.random.seed(123456789)
#      np.random.rand(5)
#      np.random.randint(0, 3, size=n)
#
#  Johansson, *Numerical Python* ch. 13, "Random Numbers", still opens this
#  way. It works, and it is the wrong API for new code, for three reasons:
#
#   1. The state is GLOBAL. Seeding it in one function changes what every
#      other function draws afterwards, including code you did not write.
#   2. Its algorithm (Mersenne Twister) and its stream are frozen for ever,
#      because changing them would break every script that relies on a seed.
#      Improvements go into the new API only.
#   3. Its argument conventions are inconsistent: `randint` excludes the top
#      value, `random_integers` included it (and was removed), `rand` takes
#      dimensions as separate arguments, `random_sample` takes a tuple.
#
#  The modern API is a Generator OBJECT:
#
#      rng = np.random.default_rng(seed)
#      rng.standard_normal((n, 2))       shape always a tuple
#      rng.integers(0, 3, size=n)        [0, 3) like range(); endpoint=True
#                                        makes the top value inclusive
#      rng.choice(x, size=..., replace=True)
#      rng.permutation(x)                a shuffled COPY
#      rng.shuffle(x)                    shuffles x IN PLACE, returns None
#
#  Pass `rng` around, or create one per function from an explicit seed. The
#  ruff rule NPY002 flags the legacy calls; this course keeps it on.
#
#  TASK
#    Rewrite the four functions on the Generator API. Read each test: they
#    check reproducibility, the inclusive top of the dice, that the caller's
#    array survives a shuffle, and that the global state is untouched.
#
#  RUN IT
#    ./npt test 04_05
#
# =============================================================================

import numpy as np


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` points in the plane and an integer label in {0, 1, 2} for each."""
    # TODO: legacy API, and it seeds the GLOBAL generator as a side effect.
    np.random.seed(seed)
    x = np.random.randn(n, 2)
    y = np.random.randint(0, 3, size=n)
    return x, y


def roll_dice(n: int, seed: int) -> np.ndarray:
    """`n` fair six-sided dice rolls, values 1 to 6 inclusive."""
    # TODO: legacy API; and randint's high is exclusive, so this die has no 6.
    np.random.seed(seed)
    return np.random.randint(1, 6, size=n)


def bootstrap_means(x: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """The mean of `resamples` resamples-with-replacement of `x`."""
    # TODO: rng.choice can draw the whole (resamples, n) matrix in one call.
    np.random.seed(seed)
    return np.array([np.random.choice(x, size=x.size).mean() for _ in range(resamples)])


def shuffled_copy(x: np.ndarray, seed: int) -> np.ndarray:
    """A shuffled copy of `x`. The caller's array is left alone."""
    # TODO: shuffle works in place -- this reorders the caller's array and
    # returns it. permutation returns a new array.
    np.random.seed(seed)
    np.random.shuffle(x)
    return x


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
