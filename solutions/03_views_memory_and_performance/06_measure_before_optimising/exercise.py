# Solution -- 03.06 Measure before optimising
import time
from collections.abc import Callable

import numpy as np


def bench(fn: Callable[[], object], repeats: int = 7, number: int = 1) -> float:
    """Seconds per call of `fn`: the best of `repeats` runs of `number` calls."""
    # One warm-up call outside the measurement: the first call pays for
    # lazy imports, cache misses, JIT compilation, allocator growth. Then the
    # MINIMUM, not the mean: everything that disturbs a run makes it slower,
    # never faster, so the fastest run is the closest to the true cost
    # (this is what timeit.repeat's documentation recommends too).
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed / number)
    return best


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
