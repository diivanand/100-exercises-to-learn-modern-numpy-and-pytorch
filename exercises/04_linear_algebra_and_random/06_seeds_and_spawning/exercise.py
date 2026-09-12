# =============================================================================
#  04.06 -- Seeds and spawning
# =============================================================================
#
#  You have a simulation to run on eight workers, and you want it
#  reproducible: the same seed gives the same result. The obvious scheme is
#
#      rng_i = np.random.default_rng(seed + i)
#
#  and it is wrong in a way that does not show up until it matters. Worker 1
#  of the run seeded with 0 is the SAME stream as worker 0 of the run seeded
#  with 1. Two experiments you believe are independent replicate each other's
#  random numbers, shifted by one worker. Average them and you have fewer
#  effective samples than you think.
#
#  NumPy's answer is `np.random.SeedSequence`. It takes an entropy value (your
#  seed) and hashes it, together with a child index, into a well-mixed
#  initial state for each child:
#
#      children = np.random.SeedSequence(seed).spawn(8)
#      rngs = [np.random.default_rng(c) for c in children]
#
#  Child 1 of seed 0 and child 0 of seed 1 now have nothing in common, and
#  every child is as independent of the others as a good generator can make
#  it. Since NumPy 1.25 a Generator can spawn directly: `rng.spawn(n)` returns
#  new generators without disturbing `rng`'s own stream, which is how you
#  hand a fresh generator to a helper that should not affect the caller.
#
#  (default_rng(None) seeds from the operating system, which is what you
#  want when you are not trying to reproduce anything.) The NumPy manual
#  page "Parallel random number generation" is the reference; Johansson
#  ch. 13 covers the distributions but predates SeedSequence.
#
#  TASK
#    Derive the worker generators by spawning, and make `fresh_child` leave
#    its parent's stream untouched.
#
#  RUN IT
#    ./npt test 04_06
#
# =============================================================================

import numpy as np


def worker_generators(seed: int, workers: int) -> list[np.random.Generator]:
    """One independent Generator per worker, all derived from `seed`."""
    # TODO: seed + i: worker i of seed s is worker i-1 of seed s+1.
    return [np.random.default_rng(seed + i) for i in range(workers)]


def simulate(seed: int, workers: int, draws: int) -> np.ndarray:
    """Each worker draws `draws` normals; returns the (workers, draws) matrix."""
    return np.stack(
        [rng.standard_normal(draws) for rng in worker_generators(seed, workers)]
    )


def fresh_child(rng: np.random.Generator) -> np.random.Generator:
    """A new generator independent of `rng`, without disturbing `rng`'s stream."""
    # TODO: drawing a seed from the parent consumes part of ITS stream, so the
    # caller's next draw is no longer what it would have been. Use rng.spawn.
    return np.random.default_rng(int(rng.integers(2**63)))


def test_simulation_is_reproducible():
    a = simulate(seed=42, workers=4, draws=1000)
    b = simulate(seed=42, workers=4, draws=1000)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 1000)


def test_workers_do_not_share_a_stream():
    a = simulate(seed=42, workers=4, draws=1000)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(a[i], a[j])
            assert abs(np.corrcoef(a[i], a[j])[0, 1]) < 0.1


def test_neighbouring_seeds_do_not_overlap():
    # The naive scheme `default_rng(seed + worker)` makes worker 1 of run 0
    # identical to worker 0 of run 1: two "different" experiments that share
    # half their random numbers. Spawning does not have that property.
    run0 = simulate(seed=0, workers=2, draws=500)
    run1 = simulate(seed=1, workers=2, draws=500)
    assert not np.array_equal(run0[1], run1[0])
    assert not np.array_equal(run0[0], run1[0])


def test_fresh_child_does_not_disturb_the_parent():
    parent = np.random.default_rng(5)
    reference = np.random.default_rng(5).standard_normal(10)
    child = fresh_child(parent)
    child_draws = child.standard_normal(10)
    np.testing.assert_array_equal(parent.standard_normal(10), reference)
    assert not np.array_equal(child_draws, reference)
