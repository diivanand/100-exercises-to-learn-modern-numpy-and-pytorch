# =============================================================================
#  10.08 -- IterableDataset
# =============================================================================
#
#  Not every dataset can be indexed. A socket, a log that is still being
#  written, a file too large to know the length of, a generator of synthetic
#  examples: these have no `__getitem__`, only a way to get the NEXT item.
#  `torch.utils.data.IterableDataset` is the Dataset for those. It has one
#  method, `__iter__`, and the DataLoader pulls from it in order, without a
#  sampler (there is nothing to sample).
#
#  That simplicity hides one trap. With `num_workers=N`, every worker holds
#  a COPY of the dataset and calls `__iter__` on it. Nothing splits the
#  stream between them. So a naive `__iter__` is replayed N times, and the
#  loader yields every record N times -- a bug that quietly multiplies your
#  epoch by N and that no shape check will catch.
#
#  The dataset has to shard itself. Inside `__iter__`,
#  `torch.utils.data.get_worker_info()` returns None in the main process
#  and, in a worker, an object with `.id` and `.num_workers`. Each worker
#  then yields only its share: for a stream of n records, worker i takes
#  records i, i + N, i + 2N, ...
#
#  TASK
#    Make `Stream.__iter__` yield only this worker's shard.
#
#  RUN IT
#    ./npt test 10_08
#
# =============================================================================

from collections.abc import Iterator

import torch
from torch.utils import data


def shard(n: int, worker_id: int, num_workers: int) -> range:
    """The slice of 0..n-1 that worker `worker_id` of `num_workers` handles."""
    # TODO: every worker gets everything.
    return range(n)


class Stream(data.IterableDataset):
    """A stream of `n` records that cannot be indexed, only iterated."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def __iter__(self) -> Iterator[torch.Tensor]:
        # TODO: ask get_worker_info() which worker this is, and iterate over
        # shard(self.n, worker_id, num_workers). In the main process the
        # answer is None: one worker, all of the data.
        for i in shard(self.n, 0, 1):
            yield torch.tensor([i, i * i])


class FakeWorkerInfo:
    def __init__(self, worker_id: int, num_workers: int) -> None:
        self.id = worker_id
        self.num_workers = num_workers


def test_shards_partition_the_range():
    for n in (0, 1, 7, 8, 100):
        for w in (1, 2, 3, 8):
            parts = [set(shard(n, i, w)) for i in range(w)]
            assert set().union(*parts) == set(range(n))
            assert sum(len(p) for p in parts) == n  # pairwise disjoint


def test_main_process_streams_everything_once():
    records = list(data.DataLoader(Stream(10), batch_size=None))
    assert [int(r[0]) for r in records] == list(range(10))
    torch.testing.assert_close(records[3], torch.tensor([3, 9]))


def test_each_worker_serves_only_its_shard(monkeypatch):
    served = []
    for worker_id in range(3):
        monkeypatch.setattr(
            data, "get_worker_info", lambda w=worker_id: FakeWorkerInfo(w, 3)
        )
        served.append([int(r[0]) for r in Stream(10)])
    assert served == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def test_batches_from_a_stream():
    loader = data.DataLoader(Stream(10), batch_size=4)
    sizes = [len(b) for b in loader]
    assert sizes == [4, 4, 2]
