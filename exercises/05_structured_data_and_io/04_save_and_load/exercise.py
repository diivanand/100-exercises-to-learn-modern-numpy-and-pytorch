# =============================================================================
#  05.04 -- Save and load
# =============================================================================
#
#  NumPy's own file format is `.npy`: a small header (shape, dtype, order)
#  followed by the raw bytes. `np.save(path, arr)` writes one, `np.load`
#  reads one, and because the header is exact, what comes back is what went
#  in, dtype included. A `.npz` is a zip of `.npy` files, one per keyword
#  argument to `np.savez` or `np.savez_compressed`, and `np.load` returns a
#  lazy mapping over its members. Johansson, *Numerical Python* ch. 18,
#  compares these with CSV, HDF5 and Parquet; for arrays between two Python
#  programs, npy/npz is the answer.
#
#  What the format does NOT do is store arbitrary Python objects. If you
#  hand np.save a dict, or an array of dtype object, it falls back to
#  pickle. Loading a pickle runs code, which is why `np.load` has refused
#  to do it by default since 2019 (`allow_pickle=False`): a file you did not
#  write yourself can then do nothing worse than fail to load. Saving a
#  dict of arrays "to keep them together" is the common way people end up
#  needing allow_pickle=True everywhere; an .npz keeps them together
#  without it.
#
#  For arrays larger than memory, `np.lib.format.open_memmap` creates a
#  .npy on disk and returns an array that reads and writes the file
#  directly through the operating system's page cache; `np.load(path,
#  mmap_mode="r")` opens an existing one the same way. McKinney ch. 12,
#  "Memory-mapped files", shows the pattern (with np.memmap, which needs
#  you to remember the shape; open_memmap stores it in the header).
#
#  TASK
#    Save and load with .npz so that loading needs no pickle, and create the
#    memmap with open_memmap.
#
#  RUN IT
#    ./npt test 05_04
#
# =============================================================================

from pathlib import Path

import numpy as np


def save_dataset(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Write `x` and `y` to one compressed .npz archive at `path`."""
    # TODO: a dict is not an array, so this is pickled. Use np.savez_compressed
    # with one keyword per array.
    np.save(path, {"x": x, "y": y}, allow_pickle=True)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the arrays back. Must not require allow_pickle."""
    # TODO: load the archive (a `with` block closes the file) and index it
    # by name.
    data = np.load(path, allow_pickle=True).item()
    return data["x"], data["y"]


def create_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    """A float32 array of the given shape backed by the file at `path`."""
    # TODO: np.lib.format.open_memmap(path, mode="w+", dtype=..., shape=...).
    return np.zeros(shape, dtype=np.float32)


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
