# Solution -- 11.01 Device-agnostic code

from typing import Any

import torch
from torch import nn


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick the device to run on.

    An explicit request wins. Otherwise use the accelerator PyTorch found at
    start-up (CUDA on the Linux box, MPS on an Apple laptop), falling back to
    the CPU so that the same code runs everywhere.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def to_device(batch: Any, device: torch.device) -> Any:
    """Move a tensor, or any nesting of tuples, lists and dicts of tensors."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list | tuple):
        return type(batch)(to_device(item, device) for item in batch)
    return batch


def run_on(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    """Evaluate `model` on `batch`, both moved to `device`, and return the output."""
    # `.to()` on a module is in place and returns the module; on a tensor it
    # returns a new tensor and leaves the original where it was. That
    # asymmetry is why the batch has to be re-bound and the model does not.
    model.to(device)
    batch = to_device(batch, device)
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
