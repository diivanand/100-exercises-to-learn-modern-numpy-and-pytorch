# Solution -- 10.07 Workers and pinning
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset


def pinning_helps() -> bool:
    """True when host memory can be pinned for a faster copy to the accelerator."""
    # Pinned (page-locked) host memory lets a CUDA copy run asynchronously by
    # DMA. On a Mac the accelerator is MPS, which shares memory with the CPU
    # and does not support pinning (the DataLoader warns and ignores it).
    if not torch.accelerator.is_available():
        return False
    return torch.accelerator.current_accelerator().type in ("cuda", "xpu")


def make_loader(dataset: TensorDataset, batch_size: int, num_workers: int) -> DataLoader:
    """A DataLoader configured sensibly for the number of workers it gets."""
    # persistent_workers and prefetch_factor only mean something when there
    # are workers; the DataLoader refuses them otherwise. pin_memory only pays
    # off when a host-to-device copy follows, and warns when it cannot help.
    kwargs: dict = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pinning_helps(),
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )


def make_dataset(n: int = 256) -> TensorDataset:
    generator = torch.Generator().manual_seed(0)
    return TensorDataset(torch.randn(n, 8, generator=generator), torch.arange(n))


def test_single_process_loader_is_valid_and_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a pin_memory warning fails the test
        loader = make_loader(make_dataset(), batch_size=32, num_workers=0)
        batches = list(loader)
    assert len(batches) == 8
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_pinning_follows_the_hardware():
    loader = make_loader(make_dataset(), batch_size=32, num_workers=0)
    assert loader.pin_memory == pinning_helps()
    xb, _ = next(iter(loader))
    if not pinning_helps():
        assert not xb.is_pinned()


def test_worker_settings_are_only_requested_with_workers():
    # Constructing with workers does not start them; iteration does. So the
    # configuration can be checked without forking anything.
    loader = make_loader(make_dataset(), batch_size=32, num_workers=2)
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 4
