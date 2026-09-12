# Solution -- 12.04 Reproducibility

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    """Seed every random source a training script draws from."""
    random.seed(seed)  # Python's own RNG: some augmentation code still uses it
    # The legacy global, deliberately: your own code uses default_rng (04.05),
    # but library code you do not control may still draw from np.random.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)  # CPU and every accelerator
    # Some CUDA kernels (scatter_add, some convolutions) pick a non-deterministic
    # algorithm by default; this makes them raise instead, so a silent source
    # of run-to-run drift becomes an error you can see.
    torch.use_deterministic_algorithms(True)


def make_loader(dataset: TensorDataset, seed: int, batch_size: int = 8) -> DataLoader:
    """A shuffling loader whose order depends on `seed` and nothing else."""
    # A DataLoader without a generator shuffles with the GLOBAL RNG, so its
    # order depends on everything that drew a random number before it. Its
    # own generator makes the order a function of `seed` alone.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_briefly(seed: int, steps: int = 20) -> torch.Tensor:
    """Train a small model and return its first-layer weights."""
    seed_everything(seed)
    x = torch.randn(64, 4)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    loader = make_loader(TensorDataset(x, y), seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step = 0
    while step < steps:
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
            step += 1
            if step == steps:
                break
    return model[0].weight.detach().clone()


def batch_order(loader: DataLoader) -> list[int]:
    return [int(i) for (xb,) in loader for i in xb]


def test_two_runs_with_the_same_seed_are_identical():
    try:
        torch.testing.assert_close(train_briefly(1), train_briefly(1), rtol=0, atol=0)
        assert not torch.equal(train_briefly(1), train_briefly(2))
    finally:
        torch.use_deterministic_algorithms(False)


def test_loader_order_depends_only_on_its_seed():
    dataset = TensorDataset(torch.arange(32))
    first = batch_order(make_loader(dataset, seed=7))
    torch.randn(1000)  # disturb the global RNG between the two loaders
    second = batch_order(make_loader(dataset, seed=7))
    assert first == second
    assert first != list(range(32))
    assert batch_order(make_loader(dataset, seed=8)) != first


def test_seed_everything_turns_on_deterministic_algorithms():
    try:
        seed_everything(0)
        assert torch.are_deterministic_algorithms_enabled()
        a = torch.rand(3)
        seed_everything(0)
        assert torch.equal(a, torch.rand(3))
        assert random.random() == (random.seed(0) or random.random())
    finally:
        torch.use_deterministic_algorithms(False)
