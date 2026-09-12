# Solution -- 09.02 nn.Module and parameters
from itertools import pairwise

import torch
from torch import nn


class Line(nn.Module):
    """y = b + w * x, with w and b registered as parameters."""

    def __init__(self) -> None:
        super().__init__()
        # nn.Parameter is a tensor that a Module registers: it appears in
        # parameters() and state_dict(), moves with .to(), and is what an
        # optimizer receives. A plain tensor attribute is none of those.
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b + self.w * x


class MLP(nn.Module):
    """A stack of Linear layers with ReLU between them."""

    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        # Submodules held in a Python list are invisible to the parent: only a
        # ModuleList (or ModuleDict, or Sequential) registers them.
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in pairwise(widths))

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
