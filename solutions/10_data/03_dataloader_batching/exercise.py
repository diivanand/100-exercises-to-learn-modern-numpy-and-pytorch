# Solution -- 10.03 DataLoader batching
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
    # Without an explicit generator the sampler draws from the global RNG,
    # so the order depends on everything that used torch.rand before it: the
    # model's initialisation, a dropout layer, another loader. Giving the
    # loader its own seeded generator makes its order a function of the seed.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )


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
