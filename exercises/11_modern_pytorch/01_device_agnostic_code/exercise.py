# =============================================================================
#  11.01 -- Device-agnostic code
# =============================================================================
#
#  The same training script has to run on three kinds of machine: the Linux
#  box with an RTX 4090 (CUDA), an Apple laptop (MPS, Metal Performance
#  Shaders), and the continuous-integration runner with nothing but a CPU.
#  Code that says `.cuda()` runs on one of the three.
#
#  Since PyTorch 2.6 there is a device-neutral namespace for the question
#  "is there an accelerator, and which one?":
#
#      torch.accelerator.is_available()          # True on CUDA, MPS, XPU, ...
#      torch.accelerator.current_accelerator()   # torch.device("cuda") / "mps"
#
#  (https://docs.pytorch.org/docs/2.14/accelerator.html). Before it, every
#  script carried its own `if torch.cuda.is_available() ... elif
#  torch.backends.mps.is_available() ...` ladder, and forgot a branch.
#
#  Two rules follow, and the second is the one people get wrong:
#
#   1. Resolve the device ONCE, at the top, and thread it through. Do not
#      sprinkle device checks through the model.
#
#   2. `module.to(device)` is IN PLACE and returns the same module.
#      `tensor.to(device)` returns a NEW tensor and leaves the original alone.
#      So `model.to(device)` on its own line is fine, but `batch.to(device)`
#      on its own line does nothing: you must write `batch = batch.to(device)`.
#      Godoy (ch. 1, "Loading Data, Devices, and CUDA") calls this out on the
#      very first page that touches a GPU, and it still catches everyone once.
#
#  Batches are rarely a single tensor. A collate function (10.04) hands you a
#  dict, or a tuple of tensors and strings. Moving "the batch" means walking
#  that structure and moving each tensor, leaving the strings and ints alone.
#
#  TASK
#    Make `resolve_device` fall back correctly, make `to_device` walk nested
#    containers, and make `run_on` move both the model and the data.
#
#  RUN IT
#    ./npt test 11_01
#
# =============================================================================

from typing import Any

import torch
from torch import nn


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick the device to run on."""
    # TODO: an explicit request should win. Otherwise ask torch.accelerator
    # whether there is an accelerator and use it; if there is none, fall back
    # to the CPU. This hard-codes CUDA, which is not what the laptop has.
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda")


def to_device(batch: Any, device: torch.device) -> Any:
    """Move a tensor, or any nesting of tuples, lists and dicts of tensors."""
    # TODO: this only handles a bare tensor. Recurse into dicts, lists and
    # tuples (keeping their types), and return non-tensors untouched.
    return batch.to(device)


def run_on(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    """Evaluate `model` on `batch`, both moved to `device`, and return the output."""
    # TODO: `batch` is not moved. `to_device` returns a new structure; the
    # result has to be re-bound. (The model line is fine: `.to` on a module
    # is in place.)
    model.to(device)
    to_device(batch, device)
    with torch.inference_mode():
        return model(batch["x"])


def test_resolve_device_prefers_an_explicit_request():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_falls_back_to_the_cpu():
    device = resolve_device()
    if torch.accelerator.is_available():
        assert device == torch.accelerator.current_accelerator()
    else:
        assert device == torch.device("cpu")


def test_to_device_walks_nested_batches():
    cpu = torch.device("cpu")
    batch = {"x": torch.zeros(2), "meta": ("id", [torch.ones(1), 3])}
    moved = to_device(batch, cpu)
    assert moved["x"].device == cpu
    assert moved["meta"][0] == "id"
    assert moved["meta"][1][0].device == cpu
    assert moved["meta"][1][1] == 3


def test_run_on_moves_model_and_data_together():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    batch = {"x": torch.randn(3, 4)}
    out = run_on(model, batch, torch.device("cpu"))
    assert out.shape == (3, 2)
    assert all(p.device == torch.device("cpu") for p in model.parameters())
    assert not out.requires_grad
