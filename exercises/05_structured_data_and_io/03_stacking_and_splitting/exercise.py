# =============================================================================
#  05.03 -- Stacking and splitting
# =============================================================================
#
#  Two joining functions, and the difference is one axis:
#
#      np.concatenate([a, b], axis=0)   join along an EXISTING axis. Two
#                                       (3, 4) arrays give (6, 4).
#      np.stack([a, b])                 join along a NEW axis. Two (3, 4)
#                                       arrays give (2, 3, 4); two (4,)
#                                       vectors give (2, 4).
#
#  np.vstack / np.hstack / np.column_stack are conveniences over these for
#  the 2-D cases; McKinney, *Python for Data Analysis* ch. 12, "Concatenating
#  and splitting arrays", tabulates them. `np.split(x, [i, j])` is the
#  inverse: it cuts at the given indices, and `np.array_split` cuts into
#  n nearly-equal pieces when the size does not divide.
#
#  The trap is `np.append`. It looks like list.append, and it is nothing
#  like it: it returns a NEW array every call (arrays cannot grow), so
#  appending in a loop copies everything each time and is quadratic; and
#  with no `axis` argument it FLATTENS both inputs first, so appending a
#  (2, 4) chunk to a (3, 4) array gives a (20,) vector. The idiom is: collect
#  the pieces in a Python list, then join once.
#
#  TASK
#    Fix `combine_chunks` (currently np.append in a loop, which the tests
#    catch twice over), and implement `batch_records` and `split_train_val`.
#
#  RUN IT
#    ./npt test 05_03
#
# =============================================================================

import numpy as np


def batch_records(records: list[np.ndarray]) -> np.ndarray:
    """Stack equal-length 1-D records into a 2-D array, one row each."""
    # TODO: this joins along the existing axis and produces one long vector.
    return np.concatenate(records)


def combine_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Join 2-D chunks (k_i, d) that arrived one at a time into one (sum k_i, d)."""
    # TODO: np.append copies the whole accumulated array on every iteration,
    # and without axis= it flattens. Collect, then concatenate once.
    out = chunks[0]
    for chunk in chunks[1:]:
        out = np.append(out, chunk)
    return out


def split_train_val(x: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the rows of `x` into a first part of `fraction` of them and the rest."""
    # TODO: np.split at one cut point.
    return x, x[:0]


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
