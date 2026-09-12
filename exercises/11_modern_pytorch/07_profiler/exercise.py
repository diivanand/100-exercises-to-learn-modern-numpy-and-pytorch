# =============================================================================
#  11.07 -- The PyTorch profiler
# =============================================================================
#
#  A timer (11.05) tells you how long a call took. A profiler tells you WHAT
#  took the time inside it: which operators, how many times, with what
#  shapes, and, on a GPU, which kernels. It is the tool for "why is this
#  training step 40 ms when the matmuls add up to 12?" -- the answer is
#  usually a data-loading stall, a `.item()` that synchronises, or a
#  hundred tiny kernels where one fused kernel was expected.
#
#      from torch.profiler import profile, ProfilerActivity, record_function
#
#      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                   record_shapes=True) as prof:
#          with record_function("train_step"):
#              loss = step(batch)
#      print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
#      prof.export_chrome_trace("trace.json")   # open in chrome://tracing or Perfetto
#
#  `activities` says which timelines to collect. CPU is the Python/ATen
#  operator stream; CUDA adds the kernels as the driver reports them, which
#  is the only way to see the true device time (Cautaerts ch. 4, "CPU time
#  versus GPU time"). The old `use_cuda=True` argument was deprecated for
#  years and REMOVED in PyTorch 2.14; code that still passes it now raises
#  TypeError (https://docs.pytorch.org/docs/2.14/profiler.html).
#
#  `record_function("name")` labels a region. Its rows in the table are
#  aggregated under that name, and in the Chrome trace it becomes a bar you
#  can read at a glance. The C++ course's chapter 17 does the same thing with
#  NVTX ranges for Nsight Systems, and in fact the PyTorch profiler emits
#  NVTX ranges too, so `nsys profile python train.py` shows both the CUDA
#  kernels and your `record_function` names on one timeline.
#
#  `key_averages()` returns one row per operator name (or per name and input
#  shape with `group_by_input_shape=True`) with `.count`, `.cpu_time_total`,
#  `.self_cpu_time_total` (time not attributed to children) and the device
#  equivalents. Rank by SELF time: total time double-counts every parent.
#
#  TASK
#    Make `profile_forward` use the current API with a named region, and make
#    `top_operators` rank by self CPU time.
#
#  RUN IT
#    ./npt test 11_07
#
# =============================================================================

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: F401


def profile_forward(model: nn.Module, x: torch.Tensor, steps: int = 3):
    """Profile `steps` forward passes and return the per-operator averages."""
    activities = [ProfilerActivity.CPU]
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator().type
        activities += {
            "cuda": [ProfilerActivity.CUDA],
            "xpu": [ProfilerActivity.XPU],
        }.get(device_type, [])
    # TODO: `use_cuda` no longer exists (removed in 2.14); the activities list
    # already says what to collect. Also wrap each forward pass in a
    # record_function("forward_pass") region so it shows up in the report.
    with profile(activities=activities, record_shapes=True, use_cuda=False) as prof:
        for _ in range(steps):
            model(x)
    return prof.key_averages()


def top_operators(averages, n: int = 5) -> list[str]:
    """Names of the `n` operators with the largest self CPU time."""
    # TODO: rank by self_cpu_time_total, descending. Total time counts every
    # parent region as well as the operator itself.
    ranked = sorted(averages, key=lambda event: event.cpu_time_total)
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
