# =============================================================================
#  10.06 -- Samplers
# =============================================================================
#
#  A DataLoader does not decide which items go in a batch; a SAMPLER does.
#  `shuffle=True` is shorthand for `sampler=RandomSampler(dataset)`. Pass
#  your own and you control the order, the subset, and the odds.
#
#  Two that come up constantly:
#
#   * `SubsetRandomSampler(indices)` iterates over a chosen subset at random.
#     Godoy uses it for the train/validation split of an image folder
#     (ch. 4, "SubsetRandomSampler").
#
#   * `WeightedRandomSampler(weights, num_samples, replacement=True)` draws
#     each item with probability proportional to its weight. This is the
#     sampling answer to class imbalance: "The class with fewer data points
#     (minority class) should get larger weights, while the class with more
#     data points (majority class) should get smaller weights. This way, on
#     average, we'll end up with mini-batches containing roughly the same
#     number of data points in each class" (Godoy ch. 4,
#     "WeightedRandomSampler"). The weight of an item is the inverse of its
#     class's count; `replacement=True` because the minority class has to be
#     drawn more often than it has members.
#
#  Both take a `generator`, for the reason 10.03 gave.
#
#  TASK
#    Make `make_balanced_sampler` weight each item by 1 / count(its class).
#
#  RUN IT
#    ./npt test 10_06
#
# =============================================================================

import torch
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
    TensorDataset,
    WeightedRandomSampler,
)


def make_balanced_sampler(labels: torch.Tensor, seed: int) -> WeightedRandomSampler:
    """Draw examples so that every class is equally likely per draw."""
    classes, counts = labels.unique(return_counts=True)
    # TODO: this weights each item by its class's COUNT, which makes the
    # majority class even more likely. Invert it.
    class_weight = counts.float()
    sample_weight = class_weight[torch.searchsorted(classes, labels)]
    return WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def make_subset_sampler(indices: list[int], seed: int) -> SubsetRandomSampler:
    """Iterate over a chosen subset of the dataset, in a random order."""
    return SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(seed))


def make_imbalanced_dataset(n: int = 1000) -> TensorDataset:
    generator = torch.Generator().manual_seed(0)
    labels = (torch.rand(n, generator=generator) < 0.1).long()  # ~10% positive
    x = torch.randn(n, 2, generator=generator)
    return TensorDataset(x, labels)


def test_balanced_sampler_evens_out_the_classes():
    ds = make_imbalanced_dataset()
    labels = ds.tensors[1]
    assert labels.float().mean() < 0.15  # the data really is imbalanced
    sampler = make_balanced_sampler(labels, seed=0)
    loader = DataLoader(ds, batch_size=50, sampler=sampler)
    seen = torch.cat([yb for _, yb in loader])
    assert len(seen) == len(ds)
    assert 0.4 < seen.float().mean().item() < 0.6


def test_balanced_sampler_is_reproducible():
    labels = make_imbalanced_dataset().tensors[1]
    a = list(make_balanced_sampler(labels, seed=1))
    b = list(make_balanced_sampler(labels, seed=1))
    assert a == b


def test_subset_sampler_stays_inside_the_subset():
    ds = make_imbalanced_dataset(100)
    subset = list(range(0, 100, 3))
    loader = DataLoader(ds, batch_size=8, sampler=make_subset_sampler(subset, seed=0))
    served = torch.cat([xb for xb, _ in loader])
    assert len(served) == len(subset)
    expected = ds.tensors[0][subset]
    # Same points, some order.
    assert torch.allclose(served.sum(dim=0), expected.sum(dim=0))
    assert not torch.equal(served, expected)
