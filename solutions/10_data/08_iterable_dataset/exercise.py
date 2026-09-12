# Solution -- 10.08 IterableDataset
from collections.abc import Iterator

import torch
from torch.utils import data


def shard(n: int, worker_id: int, num_workers: int) -> range:
    """The slice of 0..n-1 that worker `worker_id` of `num_workers` handles."""
    # Interleaved: worker 0 gets 0, w, 2w, ...; worker 1 gets 1, w+1, ...
    # Every index goes to exactly one worker, whatever n is.
    return range(worker_id, n, num_workers)


class Stream(data.IterableDataset):
    """A stream of `n` records that cannot be indexed, only iterated."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def __iter__(self) -> Iterator[torch.Tensor]:
        # In the main process get_worker_info() is None: one worker, all of
        # the data. In a worker process it says which worker this is, and the
        # dataset must serve only its shard; otherwise every worker replays
        # the whole stream and the loader yields num_workers copies of it.
        info = data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        for i in shard(self.n, worker_id, num_workers):
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
