# Solution -- 04.06 Seeds and spawning

import numpy as np


def worker_generators(seed: int, workers: int) -> list[np.random.Generator]:
    """One independent Generator per worker, all derived from `seed`."""
    # SeedSequence hashes (seed, child index) into a well-mixed state for
    # each child, so child 1 of seed 0 and child 0 of seed 1 have nothing in
    # common -- unlike default_rng(seed + i), where they are the same stream.
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(workers)]


def simulate(seed: int, workers: int, draws: int) -> np.ndarray:
    """Each worker draws `draws` normals; returns the (workers, draws) matrix."""
    # In real code each worker would run in its own process or thread with
    # its own generator. The point is that the result depends on `seed`
    # only, and each row is a statistically independent stream.
    return np.stack(
        [rng.standard_normal(draws) for rng in worker_generators(seed, workers)]
    )


def fresh_child(rng: np.random.Generator) -> np.random.Generator:
    """A new generator independent of `rng`, without disturbing `rng`'s stream."""
    # Generator.spawn (NumPy 1.25) forwards to the generator's own
    # SeedSequence, so the parent's future draws are unchanged.
    return rng.spawn(1)[0]


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
