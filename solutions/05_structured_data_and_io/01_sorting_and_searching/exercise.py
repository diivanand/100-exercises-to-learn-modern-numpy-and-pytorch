# Solution -- 05.01 Sorting and searching

import numpy as np


def leaderboard(names: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Indices that order the records by score, highest first, ties broken by
    name in alphabetical order."""
    # lexsort takes the keys LAST-KEY-FIRST: the primary key goes at the end.
    # Negating the scores turns "highest first" into an ascending sort, so
    # both keys run the same way and no reversal is needed afterwards.
    return np.lexsort((names, -scores))


def rank_in(sorted_values: np.ndarray, x: float) -> int:
    """How many entries of `sorted_values` are strictly less than `x`."""
    # searchsorted is a binary search over a sorted array: O(log n), not O(n).
    # side="left" returns the first position at which x could be inserted
    # while keeping order, which is exactly the count of strictly smaller
    # values. side="right" would also count the equal ones.
    return int(np.searchsorted(sorted_values, x, side="left"))


def top_k(values: np.ndarray, k: int) -> np.ndarray:
    """The `k` largest values, in descending order."""
    # argpartition puts the k largest at the end in O(n), unordered; sorting
    # only those k is then cheap. For k << n this beats a full sort.
    idx = np.argpartition(values, -k)[-k:]
    return np.sort(values[idx])[::-1]


def test_leaderboard_orders_by_score_then_name():
    names = np.array(["cy", "al", "bo", "di", "ed"])
    scores = np.array([90, 75, 90, 60, 75])
    order = leaderboard(names, scores)
    assert list(names[order]) == ["bo", "cy", "al", "ed", "di"]
    assert list(scores[order]) == [90, 90, 75, 75, 60]


def test_leaderboard_does_not_modify_its_inputs():
    names = np.array(["b", "a"])
    scores = np.array([1, 2])
    leaderboard(names, scores)
    assert list(names) == ["b", "a"]
    assert list(scores) == [1, 2]


def test_rank_in_counts_strictly_smaller_values():
    values = np.array([1.0, 2.0, 2.0, 5.0, 9.0])
    assert rank_in(values, 0.5) == 0
    assert rank_in(values, 2.0) == 1
    assert rank_in(values, 3.0) == 3
    assert rank_in(values, 100.0) == 5


def test_rank_in_is_logarithmic(monkeypatch):
    # A linear scan would compare against every element. searchsorted is
    # what we want, so the test insists it is used.
    called = []
    real = np.searchsorted

    def spy(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", spy)
    rank_in(np.arange(1000.0), 500.5)
    assert called, "use np.searchsorted"


def test_top_k():
    rng = np.random.default_rng(0)
    values = rng.permutation(1000).astype(float)
    np.testing.assert_array_equal(top_k(values, 3), [999.0, 998.0, 997.0])
    np.testing.assert_array_equal(top_k(values, 1), [999.0])
