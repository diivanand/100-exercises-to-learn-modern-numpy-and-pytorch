# =============================================================================
#  10.03 -- DataLoader batching
# =============================================================================
#
#  The DataLoader turns a Dataset into an iterator over mini-batches (Godoy
#  ch. 2, "DataLoader", "Mini-Batch Inner Loop"). It samples indices, fetches
#  the items, and collates them into stacked tensors:
#
#      DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
#
#  `drop_last` decides what happens to the ragged final batch (100 items in
#  batches of 32 leaves 4). Keeping it is right for evaluation, where every
#  item must be counted once; dropping it is common for training, where a
#  batch of 4 gives a noisy gradient and breaks BatchNorm.
#
#  `shuffle=True` reshuffles at the start of EVERY epoch, which is what you
#  want, and it draws the permutation from PyTorch's global random number
#  generator, which is not. The global generator is shared with parameter
#  initialisation, dropout, `torch.randn` in your own code and every other
#  loader. So the order of your batches depends on how many random numbers
#  were consumed before the loader was built, and "the same seed" gives a
#  different order when anything upstream changes. Godoy hits exactly this
#  in ch. 4 ("Seeds and more (seeds)") and reaches into the sampler to seed
#  it. The clean fix is to hand the loader its own `torch.Generator`:
#
#      DataLoader(..., shuffle=True, generator=torch.Generator().manual_seed(seed))
#
#  Now the batch order is a function of the seed and nothing else.
#
#  TASK
#    Make `make_loader` honour `drop_last` and shuffle reproducibly from
#    `seed`.
#
#  RUN IT
#    ./npt test 10_03
#
# =============================================================================

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """A DataLoader whose shuffling is reproducible from `seed` alone."""
    # TODO: the seed is never used, so the order depends on the global RNG,
    # and drop_last is ignored.
    torch.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def first_epoch_order(loader: DataLoader) -> list[int]:
    """The indices (stored as the dataset's x) in the order they were served."""
    return [int(i) for xb, _ in loader for i in xb]


def make_dataset(n: int = 100) -> TensorDataset:
    indices = torch.arange(n)
    return TensorDataset(indices, indices % 2)


def test_batches_have_the_requested_shape():
    loader = make_loader(make_dataset(100), batch_size=32, seed=0)
    sizes = [len(xb) for xb, _ in loader]
    assert sizes == [32, 32, 32, 4]


def test_drop_last_discards_the_ragged_batch():
    loader = make_loader(make_dataset(100), batch_size=32, seed=0, drop_last=True)
    assert [len(xb) for xb, _ in loader] == [32, 32, 32]


def test_shuffle_covers_every_item_once():
    loader = make_loader(make_dataset(100), batch_size=16, seed=0)
    order = first_epoch_order(loader)
    assert sorted(order) == list(range(100))
    assert order != list(range(100))


def test_same_seed_same_order_even_after_other_randomness():
    a = make_loader(make_dataset(100), batch_size=16, seed=3)
    torch.rand(1000)  # something else consumed global randomness in between
    b = make_loader(make_dataset(100), batch_size=16, seed=3)
    assert first_epoch_order(a) == first_epoch_order(b)


def test_different_seeds_different_orders():
    a = make_loader(make_dataset(100), batch_size=16, seed=3)
    b = make_loader(make_dataset(100), batch_size=16, seed=4)
    assert first_epoch_order(a) != first_epoch_order(b)


def test_each_epoch_reshuffles():
    loader = make_loader(make_dataset(100), batch_size=16, seed=0)
    assert first_epoch_order(loader) != first_epoch_order(loader)
