# =============================================================================
#  03.06 -- Measure before optimising
# =============================================================================
#
#  The last five exercises gave you levers. This one is about when to pull
#  them, and the answer is: after measuring, not before. Effective Python,
#  3rd ed., Item 92 "Profile Before Optimizing" makes the case that the
#  slow part is rarely where you think, and Item 93 introduces `timeit` for
#  the small, focused measurement that follows. Cautaerts, GPU-Accelerated
#  Computing with Python 3 and CUDA, ch. 1 opens with the same discipline
#  before touching a GPU: Amdahl's law says the speed-up you can buy is
#  bounded by the fraction of time you are actually speeding up, so find
#  that fraction first.
#
#  A timing that means something has three properties:
#
#   * WARM-UP. The first call pays for things later calls do not: imports,
#     lazily built tables, cold caches, allocator growth. Call once before
#     you start the clock.
#   * REPEATS, taking the MINIMUM. Interference from the OS, other
#     processes and the garbage collector only ever makes a run slower, so
#     the fastest of several runs is the best estimate of the true cost.
#     This is what timeit's `repeat` is for, and its documentation says the
#     same thing about the mean.
#   * A CLOCK MADE FOR IT. `time.perf_counter()` is monotonic and
#     high-resolution. `time.time()` is wall-clock and may jump.
#
#  The version below violates the first two: it times one call, and that
#  call is the first one.
#
#  TASK
#    Make `bench` warm up, repeat, and report the best per-call time.
#
#  RUN IT
#    ./npt test 03_06
#
# =============================================================================

import time
from collections.abc import Callable

import numpy as np


def bench(fn: Callable[[], object], repeats: int = 7, number: int = 1) -> float:
    """Seconds per call of `fn`: the best of `repeats` runs of `number` calls."""
    # TODO: one measurement, of the very first call, is the worst estimate
    # available. Warm up, then take the minimum over `repeats` runs, each of
    # `number` calls.
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def speedup(baseline: Callable[[], object], candidate: Callable[[], object]) -> float:
    """How many times faster `candidate` is than `baseline`."""
    return bench(baseline) / bench(candidate)


def _slow_first_call(delay: float = 0.05) -> Callable[[], None]:
    # Behaves like a function with a lazily-built table: the first call is
    # slow, every later one is fast.
    state = {"ready": False}

    def fn() -> None:
        if not state["ready"]:
            time.sleep(delay)
            state["ready"] = True

    return fn


def test_bench_ignores_the_first_call():
    per_call = bench(_slow_first_call(), repeats=3)
    assert per_call < 0.005, f"{per_call:.4f}s per call: the first call leaked in"


def test_bench_reports_per_call_time():
    def sleep_2ms() -> None:
        time.sleep(0.002)

    per_call = bench(sleep_2ms, repeats=3, number=4)
    assert 0.0015 < per_call < 0.02


def test_bench_takes_the_minimum():
    # With one very slow outlier among the repeats the estimate must not move.
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] == 4:
            time.sleep(0.05)

    assert bench(flaky, repeats=6) < 0.005


def test_speedup_orders_the_ratio():
    xs = np.random.default_rng(36).normal(size=50_000)
    ratio = speedup(lambda: sum(xs.tolist()), lambda: xs.sum())
    assert ratio > 5.0
