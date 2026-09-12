# =============================================================================
#  09.02 -- nn.Module and parameters
# =============================================================================
#
#  `nn.Module` is a container that knows what it contains. Assign an
#  `nn.Parameter` or another Module to an attribute of a Module and it is
#  REGISTERED: it shows up in `parameters()`, in `state_dict()`, it moves when
#  you call `.to(device)`, and it is what you hand to an optimiser (Godoy
#  ch. 1, "Model", "Parameters", "state_dict").
#
#  Registration happens in `__setattr__`, and only for two kinds of value:
#
#      self.w = nn.Parameter(torch.randn(1))     # registered as a parameter
#      self.fc = nn.Linear(2, 3)                 # registered as a submodule
#
#  Anything else is an ordinary Python attribute. That includes:
#
#      self.b = torch.zeros(1, requires_grad=True)   # a tensor: NOT registered
#      self.layers = [nn.Linear(2, 3), nn.Linear(3, 1)]   # a list: NOT registered
#
#  Both are silent. The tensor gets gradients, but no optimiser ever sees it,
#  so it never changes. The list's layers work in `forward`, but
#  `parameters()` is empty for them, `state_dict()` omits them, and `.to()`
#  leaves them on the CPU. Use `nn.ModuleList` (or `nn.ModuleDict`,
#  `nn.Sequential`) for collections of submodules.
#
#  TASK
#    Register every parameter and every layer in `Line` and `MLP`.
#
#  RUN IT
#    ./npt test 09_02
#
# =============================================================================

from itertools import pairwise

import torch
from torch import nn


class Line(nn.Module):
    """y = b + w * x, with w and b registered as parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        # TODO: this is a tensor attribute, not a parameter. It will never be
        # trained.
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b + self.w * x


class MLP(nn.Module):
    """A stack of Linear layers with ReLU between them."""

    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        # TODO: a Python list of modules is invisible to the parent Module.
        self.layers = [nn.Linear(a, b) for a, b in pairwise(widths)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 500) -> float:
    """Train with SGD and return the final mean squared error."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.zeros(())
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def make_data(n: int = 100, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=generator)
    y = 1.0 + 2.0 * x + 0.05 * torch.randn(n, 1, generator=generator)
    return x, y


def test_line_registers_both_parameters():
    torch.manual_seed(0)
    model = Line()
    names = sorted(name for name, _ in model.named_parameters())
    assert names == ["b", "w"]
    assert set(model.state_dict()) == {"b", "w"}
    assert all(isinstance(p, nn.Parameter) for p in model.parameters())


def test_line_learns_the_intercept():
    torch.manual_seed(0)
    x, y = make_data()
    model = Line()
    fit(model, x, y)
    assert abs(model.w.item() - 2.0) < 0.1
    # b starts at exactly 0; if it is not a parameter it stays there.
    assert abs(model.b.item() - 1.0) < 0.1


def test_mlp_registers_every_layer():
    torch.manual_seed(0)
    model = MLP([1, 8, 8, 1])
    assert sum(p.numel() for p in model.parameters()) == (1 * 8 + 8) + (8 * 8 + 8) + (
        8 * 1 + 1
    )
    assert set(model.state_dict()) == {
        f"layers.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")
    }


def test_mlp_trains_end_to_end():
    torch.manual_seed(0)
    x, y = make_data()
    model = MLP([1, 16, 1])
    before = torch.nn.functional.mse_loss(model(x), y).item()
    after = fit(model, x, y, steps=300)
    assert after < before / 10
