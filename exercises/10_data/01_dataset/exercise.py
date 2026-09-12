# =============================================================================
#  10.01 -- Dataset
# =============================================================================
#
#  `torch.utils.data.Dataset` is the smallest possible interface: `__len__`
#  and `__getitem__`. "You can think of it as a list of tuples, each tuple
#  corresponding to one point (features, label)" (Godoy ch. 2, "Dataset").
#  Nothing in it knows about batches, shuffling or workers; that is the
#  DataLoader's job (10.03), and the split is what makes both reusable.
#
#  Two rules of thumb from the same section:
#
#   * LOAD ON DEMAND. `__init__` should record what the data IS (file names,
#     a NumPy array already in memory), and `__getitem__` should do the work
#     for one item. That is what lets a DataLoader with workers parallelise
#     the loading, and what keeps a million-image dataset from being a
#     million images in RAM.
#
#   * RETURN TENSORS. The DataLoader's default collate function stacks what
#     `__getitem__` returns into a batch. It can stack tensors, and it will
#     convert NumPy arrays, but the dtype then follows NumPy: float64, which
#     a float32 model rejects at the first matmul (07.02). Convert at the
#     point of return, and choose the dtype there.
#
#  A TRANSFORM is a callable applied in `__getitem__`, so it runs per item,
#  on demand, and changing it (say, switching augmentation off for
#  evaluation) takes effect on the next fetch.
#
#  TASK
#    Give `SyntheticPoints` a `__len__`, and make `__getitem__` return
#    float32 / int64 tensors with the transform applied.
#
#  RUN IT
#    ./npt test 10_01
#
# =============================================================================

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Transform = Callable[[torch.Tensor], torch.Tensor]


class SyntheticPoints(Dataset):
    """2-D points with a label saying whether they lie inside the unit circle."""

    def __init__(self, n: int, seed: int = 0, transform: Transform | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.points = rng.uniform(-1.5, 1.5, size=(n, 2))
        self.labels = (np.linalg.norm(self.points, axis=1) < 1.0).astype(np.int64)
        self.transform = transform

    # TODO: a Dataset needs __len__; a sampler cannot draw indices without it.

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: this returns NumPy float64 and never applies the transform.
        return self.points[index], self.labels[index]


def test_len_and_items():
    ds = SyntheticPoints(50)
    assert len(ds) == 50
    x, y = ds[7]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32 and x.shape == (2,)
    assert y.dtype == torch.int64 and y.shape == ()


def test_labels_are_correct():
    ds = SyntheticPoints(200, seed=1)
    for i in range(0, 200, 20):
        x, y = ds[i]
        assert bool(y) == bool(x.norm() < 1.0)


def test_works_with_a_dataloader():
    ds = SyntheticPoints(40)
    xb, yb = next(iter(DataLoader(ds, batch_size=8)))
    assert xb.shape == (8, 2) and xb.dtype == torch.float32
    assert yb.shape == (8,) and yb.dtype == torch.int64


def test_transform_is_applied_lazily():
    ds = SyntheticPoints(10)
    x_plain, _ = ds[3]
    ds.transform = lambda x: x * 10
    x_scaled, _ = ds[3]
    torch.testing.assert_close(x_scaled, x_plain * 10)
    # The stored data was not modified: the transform ran on the way out.
    torch.testing.assert_close(
        torch.as_tensor(ds.points[3], dtype=torch.float32), x_plain
    )
