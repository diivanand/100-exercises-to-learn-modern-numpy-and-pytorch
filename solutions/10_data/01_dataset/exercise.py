# Solution -- 10.01 Dataset
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Transform = Callable[[torch.Tensor], torch.Tensor]


class SyntheticPoints(Dataset):
    """2-D points with a label saying whether they lie inside the unit circle."""

    def __init__(self, n: int, seed: int = 0, transform: Transform | None = None) -> None:
        rng = np.random.default_rng(seed)
        # The raw data can live in whatever form it arrived in; NumPy here,
        # a list of file names in real life.
        self.points = rng.uniform(-1.5, 1.5, size=(n, 2))
        self.labels = (np.linalg.norm(self.points, axis=1) < 1.0).astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Conversion happens here, per item, on demand: the DataLoader's
        # collate step stacks tensors, not NumPy arrays, and a transform is
        # applied when the item is fetched, so changing it later changes what
        # the next fetch returns.
        x = torch.as_tensor(self.points[index], dtype=torch.float32)
        y = torch.as_tensor(self.labels[index])
        if self.transform is not None:
            x = self.transform(x)
        return x, y


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
