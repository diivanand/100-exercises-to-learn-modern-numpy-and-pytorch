# Solution -- 11.07 The PyTorch profiler

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


def profile_forward(model: nn.Module, x: torch.Tensor, steps: int = 3):
    """Profile `steps` forward passes and return the per-operator averages."""
    activities = [ProfilerActivity.CPU]
    # Add the device's activity when there is one; the profiler collects the
    # kernels and their timing from the driver, which the CPU clock cannot see.
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator().type
        activities += {
            "cuda": [ProfilerActivity.CUDA],
            "xpu": [ProfilerActivity.XPU],
        }.get(device_type, [])
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(steps):
            # A named region groups everything inside it in the report,
            # exactly like an NVTX range does for Nsight Systems.
            with record_function("forward_pass"):
                model(x)
    return prof.key_averages()


def top_operators(averages, n: int = 5) -> list[str]:
    """Names of the `n` operators with the largest self CPU time."""
    ranked = sorted(averages, key=lambda event: event.self_cpu_time_total, reverse=True)
    return [event.key for event in ranked[:n]]


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 10))


def test_profile_contains_the_named_region_and_the_linear_layers():
    model = make_model()
    averages = profile_forward(model, torch.randn(32, 256))
    keys = {event.key for event in averages}
    assert "forward_pass" in keys
    assert any(key in keys for key in ("aten::linear", "aten::addmm"))


def test_the_region_was_recorded_once_per_step():
    model = make_model()
    averages = profile_forward(model, torch.randn(32, 256), steps=4)
    region = next(event for event in averages if event.key == "forward_pass")
    assert region.count == 4


def test_top_operators_are_ranked_by_self_time():
    model = make_model()
    averages = profile_forward(model, torch.randn(32, 256))
    top = top_operators(averages, n=3)
    assert len(top) == 3
    times = [next(e for e in averages if e.key == key).self_cpu_time_total for key in top]
    assert times == sorted(times, reverse=True)
    table = averages.table(sort_by="self_cpu_time_total", row_limit=5)
    assert "Self CPU" in table
