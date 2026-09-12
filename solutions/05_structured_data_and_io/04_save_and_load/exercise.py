# Solution -- 05.04 Save and load

from pathlib import Path

import numpy as np


def save_dataset(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Write `x` and `y` to one compressed .npz archive at `path`."""
    # Each keyword becomes one named array in the archive. Plain arrays only:
    # nothing here needs pickle, so nothing here needs allow_pickle to load.
    np.savez_compressed(path, x=x, y=y)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the arrays back. Must not require allow_pickle."""
    # np.load on an .npz returns a lazy mapping; the `with` closes the file.
    # allow_pickle=False is the default, and worth writing down: a pickle can
    # run arbitrary code when loaded, so never enable it for files you did
    # not write yourself.
    with np.load(path, allow_pickle=False) as archive:
        return archive["x"], archive["y"]


def create_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    """A float32 array of the given shape backed by the file at `path`."""
    # np.lib.format.open_memmap writes a proper .npy header, so the file can
    # later be opened with np.load(path, mmap_mode="r") -- or with a plain
    # np.load, which reads it all into memory.
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3)).astype(np.float32)
    y = rng.integers(0, 4, size=50)
    path = tmp_path / "data.npz"
    save_dataset(path, x, y)
    x2, y2 = load_dataset(path)
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(y2, y)
    assert x2.dtype == np.float32 and y2.dtype == y.dtype


def test_archive_loads_without_pickle(tmp_path):
    # The safe default. If the archive can only be read with allow_pickle=True
    # then object arrays (or a pickled dict) went in, which is both a security
    # problem and a portability one.
    path = tmp_path / "data.npz"
    save_dataset(path, np.arange(6.0).reshape(2, 3), np.array([1, 0]))
    with np.load(path) as archive:  # allow_pickle defaults to False
        assert sorted(archive.files) == ["x", "y"]
        assert archive["x"].shape == (2, 3)


def test_memmap_is_backed_by_the_file(tmp_path):
    path = tmp_path / "big.npy"
    m = create_memmap(path, (1000, 8))
    assert isinstance(m, np.memmap)
    assert m.dtype == np.float32 and m.shape == (1000, 8)
    m[:] = 0.0
    m[999, 7] = 42.0
    m.flush()
    del m
    reloaded = np.load(path, mmap_mode="r")
    assert reloaded[999, 7] == 42.0
    assert reloaded.sum() == 42.0
