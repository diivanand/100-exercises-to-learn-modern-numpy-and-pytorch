# Solution -- 05.03 Stacking and splitting

import numpy as np


def batch_records(records: list[np.ndarray]) -> np.ndarray:
    """Stack equal-length 1-D records into a 2-D array, one row each."""
    # np.stack joins along a NEW axis; np.concatenate joins along an existing
    # one. Rows from vectors is the new-axis case.
    return np.stack(records)


def combine_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Join 2-D chunks (k_i, d) that arrived one at a time into one (sum k_i, d)."""
    # Collect, then join ONCE. Appending in a loop copies the growing array
    # every time (quadratic), and np.append with no axis flattens its inputs.
    return np.concatenate(chunks, axis=0)


def split_train_val(x: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the rows of `x` into a first part of `fraction` of them and the rest."""
    cut = round(fraction * x.shape[0])
    train, val = np.split(x, [cut])
    return train, val


def test_batch_records_adds_a_row_axis():
    records = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    batch = batch_records(records)
    assert batch.shape == (3, 2)
    np.testing.assert_array_equal(batch[1], [3.0, 4.0])


def test_combine_chunks_keeps_the_column_axis():
    rng = np.random.default_rng(0)
    chunks = [rng.standard_normal((k, 4)) for k in (3, 1, 5)]
    out = combine_chunks(chunks)
    assert out.shape == (9, 4)
    np.testing.assert_array_equal(out[3], chunks[1][0])
    np.testing.assert_array_equal(out[4:], chunks[2])


def test_combine_chunks_is_linear_time(monkeypatch):
    # Joining in a loop copies the accumulated array on every step. The test
    # counts how many times a joining function is called: once is right.
    calls = []
    real = np.concatenate

    def counting(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", counting)
    chunks = [np.ones((2, 3)) for _ in range(50)]
    assert combine_chunks(chunks).shape == (100, 3)
    assert len(calls) == 1, "join the chunks with one concatenate"


def test_split_train_val():
    x = np.arange(20).reshape(10, 2)
    train, val = split_train_val(x, 0.7)
    assert train.shape == (7, 2) and val.shape == (3, 2)
    np.testing.assert_array_equal(np.concatenate([train, val]), x)
