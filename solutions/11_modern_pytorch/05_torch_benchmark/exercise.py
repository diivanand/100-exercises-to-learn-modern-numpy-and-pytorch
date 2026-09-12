# Solution -- 11.05 torch.utils.benchmark

from collections.abc import Callable

import torch
import torch.utils.benchmark as benchmark


def time_call(
    fn: Callable[..., torch.Tensor], *args: torch.Tensor, min_run_time: float = 0.05
) -> benchmark.Measurement:
    """Time `fn(*args)` properly and return the Measurement."""
    # Timer does the four things a hand-rolled loop forgets: it warms up, it
    # synchronises the device before reading the clock, it runs enough
    # iterations to fill `min_run_time`, and it reports a distribution
    # (median, IQR) rather than one number.
    timer = benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        label=getattr(fn, "__name__", "fn"),
    )
    return timer.blocked_autorange(min_run_time=min_run_time)


def compare(
    candidates: dict[str, Callable[..., torch.Tensor]], *args: torch.Tensor
) -> dict[str, float]:
    """Median seconds per call for each candidate, fastest first."""
    medians = {name: time_call(fn, *args).median for name, fn in candidates.items()}
    return dict(sorted(medians.items(), key=lambda item: item[1]))


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
