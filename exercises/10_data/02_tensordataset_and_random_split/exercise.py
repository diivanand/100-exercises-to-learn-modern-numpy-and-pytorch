# =============================================================================
#  10.02 -- TensorDataset and random_split
# =============================================================================
#
#  When the whole dataset is already two tensors, there is nothing to write:
#  `TensorDataset(x, y)` is the Dataset from 10.01, done (Godoy ch. 2,
#  "TensorDataset").
#
#  Splitting it into training and validation sets is where the trouble is.
#  Data arrives ORDERED far more often than it arrives shuffled: by class
#  (all the cats, then all the dogs), by time, by which sensor wrote the file.
#  Cut such data with a slice and the validation set is not a sample of the
#  problem; it is the last class, or the last month. The model is then
#  scored on something it never saw an example of, or on a distribution
#  shift you did not intend to test.
#
#  `torch.utils.data.random_split(dataset, [n_train, n_val], generator=...)`
#  permutes the indices first and returns two `Subset` objects that share the
#  underlying data (Godoy ch. 2, "Random Split"). The `generator` argument
#  is not optional in any code you want to run twice: without it the split
#  depends on the global RNG state, and so on everything that happened
#  before it (12.04).
#
#  TASK
#    Split at random, reproducibly.
#
#  RUN IT
#    ./npt test 10_02
#
# =============================================================================

import torch
from torch.utils.data import Subset, TensorDataset


def make_splits(
    x: torch.Tensor, y: torch.Tensor, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    """Split (x, y) into a training and a validation dataset, at random."""
    dataset = TensorDataset(x, y)
    n_val = round(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    # TODO: the last n_val rows are whatever the data happens to end with.
    # Here that is one class, all of it.
    train_ds = Subset(dataset, list(range(n_train)))
    val_ds = Subset(dataset, list(range(n_train, n_train + n_val)))
    return train_ds, val_ds


def make_sorted_data(n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Data as it often arrives: every class 0 example before every class 1."""
    generator = torch.Generator().manual_seed(0)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n - n // 2)]).long()
    x = torch.randn(n, 3, generator=generator) + y[:, None].float() * 2.0
    return x, y


def test_sizes():
    x, y = make_sorted_data(100)
    train_ds, val_ds = make_splits(x, y, val_fraction=0.2, seed=0)
    assert len(train_ds) == 80 and len(val_ds) == 20


def test_both_classes_appear_in_validation():
    x, y = make_sorted_data(100)
    _, val_ds = make_splits(x, y, val_fraction=0.2, seed=0)
    labels = torch.stack([label for _, label in val_ds])
    assert set(labels.tolist()) == {0, 1}


def test_split_is_disjoint_and_exhaustive():
    x, y = make_sorted_data(100)
    train_ds, val_ds = make_splits(x, y, val_fraction=0.3, seed=0)
    train_idx, val_idx = set(train_ds.indices), set(val_ds.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(100))


def test_split_is_reproducible_with_a_seed_and_differs_without():
    x, y = make_sorted_data(100)
    a, _ = make_splits(x, y, 0.2, seed=7)
    b, _ = make_splits(x, y, 0.2, seed=7)
    c, _ = make_splits(x, y, 0.2, seed=8)
    assert list(a.indices) == list(b.indices)
    assert list(a.indices) != list(c.indices)


def test_items_are_tensor_pairs():
    x, y = make_sorted_data(10)
    train_ds, _ = make_splits(x, y, 0.2, seed=0)
    xi, yi = train_ds[0]
    assert xi.shape == (3,) and yi.shape == ()
