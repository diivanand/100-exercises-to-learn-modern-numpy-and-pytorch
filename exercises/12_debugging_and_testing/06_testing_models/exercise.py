# =============================================================================
#  12.06 -- Testing models
# =============================================================================
#
#  A model has no `assert result == 42`. What it has is a handful of
#  properties that every trainable model must satisfy, and a small number of
#  bugs that account for most silent failures. Four tests, each of which
#  takes under a second on the CPU, catch nearly all of them:
#
#   1. OUTPUT SHAPE, parametrised over a few sizes (`pytest.mark.parametrize`
#      is the tool; Effective Python, 3rd ed., Items 108-109 argue for
#      verifying related behaviours together and preferring tests that
#      exercise the whole thing). This catches the squeeze that drops the
#      batch dimension when it is 1 (07.08).
#
#   2. EVERY PARAMETER MOVES after one optimiser step: `p.grad` is not None
#      and the value changed. This catches a layer that is defined but not
#      used, a `.detach()` left in the forward pass from evaluation code, a
#      frozen parameter nobody meant to freeze, and `requires_grad=False`
#      on a tensor that should have been an nn.Parameter (09.02).
#
#   3. OVERFIT ONE BATCH. Train on 16 examples until the loss is nearly
#      zero. A correct model memorises them in a few hundred steps; one
#      that cannot has a bug in the loss (09.04), the labels, or the data
#      pipeline -- and you found it before spending a night on the 4090.
#      Karpathy's "A Recipe for Training Neural Networks" calls this the
#      single most useful sanity check.
#
#   4. EVAL MODE IS DETERMINISTIC and tracks no gradients (09.07): two
#      forward passes agree exactly, and the output has requires_grad False
#      under inference_mode.
#
#  The model below fails two of them. The bugs are the ordinary kind: a
#  `.detach()` copied from an evaluation snippet, and a layer that was
#  added to `__init__` but never wired into `forward`.
#
#  TASK
#    Fix `TinyClassifier` so that the tests pass, and complete
#    `parameters_that_did_not_change` so that it also reports parameters
#    with no gradient.
#
#  RUN IT
#    ./npt test 12_06
#
# =============================================================================

import pytest
import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, features: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.GELU())
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: two bugs. The detach() cuts the encoder off from the loss,
        # so its parameters never train; and self.norm is never applied.
        h = self.encoder(x).detach()
        return self.head(h)


def overfit_one_batch(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 200, lr: float = 1e-2
) -> float:
    """Train on one batch only and return the final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss = torch.tensor(float("inf"))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()


def parameters_that_did_not_change(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> list[str]:
    """Names of parameters unchanged (or without a gradient) after one step."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()
    stuck = []
    for name, p in model.named_parameters():
        # TODO: a parameter with p.grad None never took part in the loss;
        # report it too.
        if torch.equal(p.detach(), before[name]):
            stuck.append(name)
    return stuck


def make_batch(features: int, classes: int, n: int = 32):
    torch.manual_seed(0)
    return torch.randn(n, features), torch.randint(0, classes, (n,))


@pytest.mark.parametrize(("features", "classes"), [(4, 2), (16, 5)])
def test_output_shape(features: int, classes: int):
    torch.manual_seed(0)
    model = TinyClassifier(features, hidden=8, classes=classes)
    x, _ = make_batch(features, classes)
    assert model(x).shape == (32, classes)


def test_every_parameter_receives_a_gradient_and_moves():
    torch.manual_seed(0)
    model = TinyClassifier(4, hidden=8, classes=3)
    x, y = make_batch(4, 3)
    assert parameters_that_did_not_change(model, x, y) == []


def test_the_model_can_overfit_one_batch():
    torch.manual_seed(0)
    model = TinyClassifier(4, hidden=32, classes=3)
    x, y = make_batch(4, 3, n=16)
    final = overfit_one_batch(model, x, y)
    assert final < 0.05, f"could not memorise 16 examples: loss {final:.3f}"


def test_eval_mode_is_deterministic_and_does_not_track_gradients():
    torch.manual_seed(0)
    model = TinyClassifier(4, hidden=8, classes=3).eval()
    x, _ = make_batch(4, 3)
    with torch.inference_mode():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)
    assert not a.requires_grad
