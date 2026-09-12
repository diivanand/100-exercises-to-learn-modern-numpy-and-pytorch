# =============================================================================
#  12.05 -- Hooks
# =============================================================================
#
#  A model is a black box only if you let it be. Hooks are callbacks that
#  PyTorch runs at fixed points of the forward and backward passes, and they
#  let you look inside a model without editing it -- which matters when the
#  model is torchvision's, or a colleague's, or already deployed.
#
#      handle = layer.register_forward_hook(fn)     # fn(module, inputs, output)
#      handle = layer.register_full_backward_hook(fn)  # fn(module, grad_in, grad_out)
#      handle = tensor.register_hook(fn)            # fn(grad) -> grad or None
#
#  Godoy uses forward hooks to capture the activations of every layer for
#  plotting (ch. 5, the section on hooks); the same code answers "which
#  layer's outputs blew up?" and "did dropout actually change anything?".
#  A tensor hook that returns a replacement gradient is how gradient
#  clipping at one point in the graph is done, and how you would inject a
#  nan on purpose to test 12.02's detector.
#
#  Two rules, both about lifetimes:
#
#   1. KEEP THE HANDLE AND REMOVE THE HOOK. A hook lives as long as the
#      module. Register one in a loop, or forget to remove it after the
#      experiment, and every later forward pass runs it too: your recorder
#      keeps growing, and if it stored the outputs with their graph it
#      keeps every activation of every batch alive. A context manager that
#      registers on `__enter__` and removes on `__exit__` makes this
#      impossible to get wrong.
#
#   2. `.detach()` WHAT YOU STORE. The output handed to a forward hook is
#      part of the autograd graph. Storing it as-is keeps the whole graph of
#      that forward pass in memory until your list is cleared.
#
#  TASK
#    Make `ActivationRecorder` remove its hooks on exit and store detached
#    statistics, and make `clip_gradient_at` return the clamped gradient
#    from its hook.
#
#  RUN IT
#    ./npt test 12_05
#
# =============================================================================

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
            # TODO: `output` is attached to the graph. Detach before you
            # compute anything you keep.
            self.stats[name].append((output.mean(), output.std()))

        return hook

    def __enter__(self) -> ActivationRecorder:
        modules = dict(self.model.named_modules())
        for name in self.names:
            # TODO: register_forward_hook returns a handle. Keep it in
            # self._handles, or the hook can never be removed.
            modules[name].register_forward_hook(self._make_hook(name))
        return self

    def __exit__(self, *exc) -> None:
        # TODO: remove every handle.
        pass


def clip_gradient_at(t: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp the gradient that flows back through `t` to [-limit, limit]."""
    # TODO: a tensor hook that returns None leaves the gradient unchanged.
    # Return the clamped gradient.
    t.register_hook(lambda grad: grad.clamp(-limit, limit) and None)
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
