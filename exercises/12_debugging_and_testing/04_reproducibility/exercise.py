# =============================================================================
#  12.04 -- Reproducibility
# =============================================================================
#
#  "It worked yesterday" is not a bug report. A training run you cannot
#  repeat is a run you cannot debug, and the difference between two runs
#  that should be identical is the first thing to check when a change
#  "made things worse". Godoy spends a section on it (ch. 4, "Seeds and more
#  (seeds)") and ends up with a `set_seed` that seeds four things, because
#  a PyTorch program draws random numbers from more places than it looks:
#
#      random.seed(seed)         Python's RNG -- augmentation code, shuffles
#      np.random.seed(seed)      NumPy's legacy global -- third-party code
#      torch.manual_seed(seed)   PyTorch, CPU and every accelerator at once
#
#  and, separately,
#
#      torch.use_deterministic_algorithms(True)
#
#  because seeding is not enough on a GPU: some kernels (scatter/index_add,
#  some cuDNN convolutions) are non-deterministic by design -- atomic adds
#  in a different order each run -- and this switch makes them either use a
#  deterministic algorithm or raise. (Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in
#  the environment for the cuBLAS ones; the error message tells you.)
#
#  The DataLoader is the subtle one. `shuffle=True` draws its permutation
#  from the GLOBAL torch RNG unless you give it its own:
#
#      DataLoader(ds, shuffle=True, generator=torch.Generator().manual_seed(seed))
#
#  Without that, the order of the batches depends on how many random numbers
#  were drawn before the loader was created -- initialising one more layer
#  changes the data order, and now two runs differ for a reason that has
#  nothing to do with the change you made. With `num_workers > 0` there is a
#  further seed per worker, `worker_init_fn`, which 10.07 covers.
#
#  Reproducibility has a price: deterministic kernels are slower, and
#  `cudnn.benchmark` (11.06) must be off. Turn it on to debug and to publish;
#  measure with it off.
#
#  TASK
#    Complete `seed_everything` and give the DataLoader its own generator.
#
#  RUN IT
#    ./npt test 12_04
#
# =============================================================================

import random

import numpy as np  # noqa: F401
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    """Seed every random source a training script draws from."""
    # TODO: this seeds torch only. Python's `random`, NumPy's legacy global
    # RNG and the deterministic-algorithms switch are missing.
    torch.manual_seed(seed)


def make_loader(dataset: TensorDataset, seed: int, batch_size: int = 8) -> DataLoader:
    """A shuffling loader whose order depends on `seed` and nothing else."""
    # TODO: without a generator the shuffle uses the global RNG, so the
    # order depends on everything drawn before this call. Pass
    # generator=torch.Generator().manual_seed(seed).
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
