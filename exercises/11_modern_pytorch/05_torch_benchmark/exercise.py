# =============================================================================
#  11.05 -- torch.utils.benchmark
# =============================================================================
#
#  "Profile before optimising" (Effective Python, 3rd ed., Item 92) and
#  "use timeit for microbenchmarks" (Item 93) are the right instincts, but a
#  plain `time.perf_counter()` around a PyTorch call is wrong on a GPU and
#  misleading on a CPU, for four reasons:
#
#   1. GPU work is asynchronous. A kernel launch returns to Python before the
#      kernel runs, so the clock measures the launch, not the work. Cautaerts
#      (ch. 4, "CPU time versus GPU time") puts it as: "the CPU's perception of
#      elapsed time does not necessarily represent the actual time the GPU
#      spent executing a kernel". You must synchronise before reading the
#      clock -- and the same is true of MPS on a Mac.
#   2. The first call is not representative: it pays for lazy initialisation,
#      cuBLAS handle creation, the compile of a Triton kernel.
#   3. One run is not a measurement. Timings have a distribution, and the
#      number you want is the median, with an interquartile range to say how
#      much to trust it.
#   4. Sub-millisecond calls need many iterations to be measured at all.
#
#  `torch.utils.benchmark.Timer` does all four. It has the same shape as
#  `timeit.Timer` (a `stmt` string and a `globals` dict), synchronises the
#  device around the block, warms up, and its `blocked_autorange(min_run_time)`
#  picks the iteration count for you and returns a `Measurement` with
#  `.median`, `.mean`, `.iqr`, `.times` and `.number_per_run`.
#  (https://docs.pytorch.org/docs/2.14/benchmark_utils.html)
#
#  Whole-program questions ("where does the time go?") are for a profiler,
#  not a timer: 11.07 for the operator view, and on the 4090
#  `nsys profile python train.py` for the timeline (Cautaerts ch. 4,
#  "Profiling GPU timeline with Nsight Systems"; the C++ course's chapter 17
#  uses the same tool).
#
#  TASK
#    Rewrite `time_call` on top of benchmark.Timer, and make `compare` report
#    medians ordered fastest first.
#
#  RUN IT
#    ./npt test 11_05
#
# =============================================================================

import time
from collections.abc import Callable

import torch
import torch.utils.benchmark as benchmark


def time_call(
    fn: Callable[..., torch.Tensor], *args: torch.Tensor, min_run_time: float = 0.05
) -> benchmark.Measurement:
    """Time `fn(*args)` properly and return the Measurement."""
    # TODO: one cold call on a wall clock, with no synchronisation and no
    # distribution. Build a benchmark.Timer with stmt="fn(*args)" and a
    # globals dict, and return timer.blocked_autorange(min_run_time=...).
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start  # type: ignore[return-value]


def compare(
    candidates: dict[str, Callable[..., torch.Tensor]], *args: torch.Tensor
) -> dict[str, float]:
    """Median seconds per call for each candidate, fastest first."""
    # TODO: use the Measurement's .median and sort by it.
    return {name: time_call(fn, *args) for name, fn in candidates.items()}


def matmul_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        out[i] = a[i] @ b
    return out


def matmul_fused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def test_time_call_returns_a_measurement_with_a_median():
    a = torch.randn(64, 64)
    m = time_call(matmul_fused, a, a)
    assert isinstance(m, benchmark.Measurement)
    assert m.median > 0
    assert m.number_per_run >= 1


def test_measurement_is_a_distribution_not_a_single_number():
    a = torch.randn(64, 64)
    m = time_call(matmul_fused, a, a)
    # blocked_autorange runs several blocks; each contributes a time.
    assert len(m.times) >= 2
    assert m.iqr >= 0


def test_compare_orders_candidates_fastest_first():
    torch.manual_seed(0)
    a, b = torch.randn(256, 64), torch.randn(64, 64)
    torch.testing.assert_close(matmul_loop(a, b), matmul_fused(a, b))
    result = compare({"loop": matmul_loop, "fused": matmul_fused}, a, b)
    assert list(result) == ["fused", "loop"]
    assert result["loop"] > 5 * result["fused"]
