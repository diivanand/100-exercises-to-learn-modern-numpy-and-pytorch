# =============================================================================
#  10.07 -- Workers and pinning
# =============================================================================
#
#  On a GPU machine the model is rarely the bottleneck; the loader is. Four
#  DataLoader arguments decide whether the GPU waits for data:
#
#   * `num_workers=N` fetches items in N subprocesses, so decoding and
#     augmentation run in parallel with the training step. `0` means "in the
#     main process", which is what you want while debugging and what this
#     test uses. On Linux the workers are forked; on macOS and Windows they
#     are SPAWNED, so the script is re-imported in each worker and anything
#     at module level runs N extra times -- hence the `if __name__ ==
#     "__main__":` guard around training scripts.
#
#   * `persistent_workers=True` keeps the workers alive between epochs
#     instead of respawning them; `prefetch_factor=k` lets each worker run k
#     batches ahead. Both only make sense with workers, and the DataLoader
#     raises a ValueError if you ask for either with `num_workers=0`.
#
#   * `pin_memory=True` puts each batch in page-locked host memory, which is
#     what lets a CUDA copy run asynchronously (`.to(device,
#     non_blocking=True)`). It costs nothing on the GPU box and it is worth
#     a real fraction of the step time. On a machine with no such device it
#     does nothing and PyTorch says so, once, in a warning. On a Mac the
#     accelerator is MPS, which shares the CPU's memory and does not support
#     pinning at all.
#
#  So the right configuration depends on the machine, and a loader factory
#  should decide from `num_workers` and the hardware rather than hard-coding
#  values that are correct only on the author's workstation.
#
#  TASK
#    Make `make_loader` valid for `num_workers=0`, quiet on a machine without
#    a CUDA device, and fully tuned when workers are requested.
#
#  RUN IT
#    ./npt test 10_07
#
#  ON THE GPU MACHINE
#    Try `num_workers=4` and time an epoch with and without `pin_memory`; the
#    difference is the copy the GPU no longer waits for.
#
# =============================================================================

import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset


def pinning_helps() -> bool:
    """True when host memory can be pinned for a faster copy to the accelerator."""
    # TODO: MPS counts as an accelerator but cannot use pinned memory. Only
    # CUDA (and XPU) can.
    return torch.accelerator.is_available()


def make_loader(dataset: TensorDataset, batch_size: int, num_workers: int) -> DataLoader:
    """A DataLoader configured sensibly for the number of workers it gets."""
    # TODO: persistent_workers and prefetch_factor are only valid with
    # workers; pin_memory only helps (and is only quiet) with a CUDA device.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(0),
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
