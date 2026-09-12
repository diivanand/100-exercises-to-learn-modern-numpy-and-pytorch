# Solution -- 12.05 Hooks

from __future__ import annotations

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ActivationRecorder:
    """Record the mean and std of every named submodule's output.

    Use as a context manager: the hooks exist only inside the `with` block.
    """

    def __init__(self, model: nn.Module, names: list[str]) -> None:
        self.model = model
        self.names = names
        self.stats: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
        self._handles: list[RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            # detach(): a recorded activation must not keep the graph alive.
            out = output.detach()
            self.stats[name].append((out.mean().item(), out.std().item()))

        return hook

    def __enter__(self) -> ActivationRecorder:
        modules = dict(self.model.named_modules())
        for name in self.names:
            # register_forward_hook returns a handle; keeping it is the only
            # way to remove the hook later (Godoy ch. 5, on hooks).
            self._handles.append(
                modules[name].register_forward_hook(self._make_hook(name))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def clip_gradient_at(t: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp the gradient that flows back through `t` to [-limit, limit]."""
    # A tensor hook runs when t.grad is computed and may return a replacement.
    t.register_hook(lambda grad: grad.clamp(-limit, limit))
    return t


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def test_recorder_captures_stats_for_each_named_layer():
    model = make_model()
    with ActivationRecorder(model, ["0", "2"]) as rec:
        model(torch.randn(16, 4))
        model(torch.randn(16, 4))
    assert len(rec.stats["0"]) == 2
    assert len(rec.stats["2"]) == 2
    mean, std = rec.stats["0"][0]
    assert isinstance(mean, float) and std > 0


def test_hooks_are_removed_on_exit():
    model = make_model()
    with ActivationRecorder(model, ["0"]) as rec:
        model(torch.randn(16, 4))
    model(torch.randn(16, 4))  # outside the block: must not be recorded
    assert len(rec.stats["0"]) == 1
    assert len(model[0]._forward_hooks) == 0


def test_recorded_activations_do_not_hold_the_graph():
    model = make_model()
    x = torch.randn(16, 4, requires_grad=True)
    with ActivationRecorder(model, ["0"]):
        out = model(x)
    out.sum().backward()
    assert x.grad is not None


def test_gradient_hook_clips_what_flows_back():
    w = torch.tensor([3.0], requires_grad=True)
    h = clip_gradient_at(w * 1.0, limit=0.5)
    (h * 100.0).sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([0.5]))
