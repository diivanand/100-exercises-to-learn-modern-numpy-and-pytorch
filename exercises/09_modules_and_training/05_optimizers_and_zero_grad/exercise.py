# =============================================================================
#  09.05 -- Optimisers and zero_grad
# =============================================================================
#
#  An optimiser owns the update rule (Godoy ch. 1, "Optimizer"; Kneusel,
#  *Math for Deep Learning*, ch. 11 compares the rules). You give it the
#  parameters once; after that `step()` updates every one of them from its
#  `.grad`, and `zero_grad()` clears the gradients for the next round.
#
#      optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#      loss.backward()
#      optimizer.step()
#      optimizer.zero_grad()
#
#  Two things about that loop.
#
#  GRADIENTS ACCUMULATE. `backward()` adds into `.grad`. That is deliberate
#  (09.10 uses it to sum micro-batches), but it means a loop that never calls
#  `zero_grad()` feeds each step the sum of every gradient so far. With
#  momentum, the effect compounds: the update grows with every step and the
#  loss diverges. Since 2.0, `zero_grad()` sets `.grad` to None instead of
#  filling it with zeros (`set_to_none=True` is the default): cheaper, and
#  the next `backward()` allocates afresh.
#
#  PARAMETER GROUPS. The optimiser takes either an iterable of parameters or
#  a list of dicts, each with its own `params` and its own overrides:
#
#      torch.optim.AdamW([
#          {"params": weights, "weight_decay": 0.01},
#          {"params": biases,  "weight_decay": 0.0},
#      ], lr=1e-3)
#
#  Weight decay pulls parameters towards zero. That is regularisation for a
#  weight matrix and pure harm for a bias or a normalisation scale, so the
#  usual practice is to decay only tensors with more than one dimension.
#
#  ADAM VS ADAMW. Adam with `weight_decay` adds an L2 term to the GRADIENT
#  before the adaptive scaling, which then largely undoes it. AdamW applies
#  the decay to the PARAMETER directly ("decoupled"). Use AdamW.
#
#  TASK
#    Build the two-group AdamW optimiser, and clear the gradients between
#    steps in `train_steps`.
#
#  RUN IT
#    ./npt test 09_05
#
# =============================================================================

import torch
from torch import nn


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay on weights only, never on biases."""
    # TODO: Adam with weight_decay is L2 on the gradient, not decoupled decay,
    # and this decays the biases too. Use AdamW with two parameter groups.
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
) -> list[float]:
    """Run `steps` full-batch updates and return the loss after each."""
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        # TODO: the gradients from this step are still in .grad when the next
        # backward() runs, and are added to.
        losses.append(loss.item())
    return losses


def make_data(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    true_w = torch.tensor([[1.0], [-2.0], [0.5]])
    y = x @ true_w + 0.3
    return x, y


def test_parameter_groups_exempt_biases():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    decayed = {
        id(p)
        for g in optimizer.param_groups
        if g["weight_decay"] > 0
        for p in g["params"]
    }
    for name, p in model.named_parameters():
        assert (id(p) in decayed) == name.endswith("weight"), name
    # Every parameter is in exactly one group.
    assert sum(len(g["params"]) for g in optimizer.param_groups) == 4


def test_momentum_sgd_converges_on_a_convex_problem():
    torch.manual_seed(0)
    x, y = make_data()
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = train_steps(model, optimizer, x, y, steps=200)
    assert losses[-1] < 1e-4
    torch.testing.assert_close(
        model.weight.detach(), torch.tensor([[1.0, -2.0, 0.5]]), atol=0.02, rtol=0
    )


def test_gradients_are_cleared_between_steps():
    torch.manual_seed(0)
    x, y = make_data()
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_steps(model, optimizer, x, y, steps=1)
    # zero_grad(set_to_none=True) is the default since 2.0: the tensor is gone,
    # not merely zeroed.
    assert all(p.grad is None for p in model.parameters())


def test_stale_gradients_would_derail_adamw():
    torch.manual_seed(0)
    x, y = make_data()
    model = nn.Linear(3, 1)
    optimizer = make_optimizer(model, lr=0.05, weight_decay=0.0)
    losses = train_steps(model, optimizer, x, y, steps=300)
    assert losses[-1] < 1e-3
