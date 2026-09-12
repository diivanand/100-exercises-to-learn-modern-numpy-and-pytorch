# Solution -- 09.05 Optimizers and zero_grad
import torch
from torch import nn


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay on weights only, never on biases."""
    # Two parameter groups: the second overrides weight_decay for the
    # one-dimensional parameters (biases, norm scales), which should not be
    # pulled towards zero.
    decay = [p for name, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for name, p in model.named_parameters() if p.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


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
        # Gradients accumulate across backward() calls by design (that is how
        # 09.10 sums micro-batches). Between steps they must be cleared, and
        # zero_grad() now sets them to None rather than filling with zeros.
        optimizer.zero_grad()
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
