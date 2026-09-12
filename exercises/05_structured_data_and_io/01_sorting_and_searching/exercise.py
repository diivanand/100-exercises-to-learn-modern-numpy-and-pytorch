# =============================================================================
#  05.01 -- Sorting and searching
# =============================================================================
#
#  `a.sort()` sorts in place and returns None; `np.sort(a)` returns a sorted
#  copy. Neither is what you want when the array you are sorting is one
#  column of a record: sorting the scores loses which name each score
#  belonged to. The tool for that is an INDIRECT sort, `np.argsort`, which
#  returns the permutation that would sort the array, so you can apply the
#  same permutation to every other column. McKinney, *Python for Data
#  Analysis* ch. 12, "Indirect sorts: argsort and lexsort", is the classic
#  treatment (the API has not changed since; his `randn` has, see 04.05).
#
#  For several keys, `np.lexsort((secondary, primary))` sorts by the LAST key
#  first. The argument order surprises everyone once. For a descending
#  numeric key, negate it rather than reversing the result: reversing also
#  reverses the tie-break order.
#
#  Searching a SORTED array is a binary search, `np.searchsorted`, O(log n).
#  Effective Python (3rd ed.), Item 102, makes the same point about the
#  standard library's `bisect`; searchsorted is the vectorised version and
#  takes an array of needles. `side="left"` gives the position of the first
#  value >= x, `side="right"` the first value > x, and the difference between
#  them is the count of exact matches.
#
#  Finally, when you only need the k largest, a full sort is wasteful:
#  `np.argpartition(a, -k)[-k:]` finds them in O(n) (unordered), and you sort
#  those k.
#
#  TASK
#    Fix `leaderboard` (it currently corrupts its inputs and loses the
#    names), make `rank_in` a binary search, and implement `top_k`.
#
#  RUN IT
#    ./npt test 05_01
#
# =============================================================================

import numpy as np


def leaderboard(names: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Indices that order the records by score, highest first, ties broken by
    name in alphabetical order."""
    # TODO: this sorts the caller's array in place (returning None), then
    # builds indices that no longer correspond to the names. Use np.lexsort
    # with the primary key last.
    scores.sort()
    return np.arange(len(scores))[::-1]


def rank_in(sorted_values: np.ndarray, x: float) -> int:
    """How many entries of `sorted_values` are strictly less than `x`."""
    # TODO: a linear scan over a sorted array. Use np.searchsorted.
    count = 0
    for value in sorted_values:
        if value < x:
            count += 1
    return count


def top_k(values: np.ndarray, k: int) -> np.ndarray:
    """The `k` largest values, in descending order."""
    # TODO: np.argpartition, then sort only the k survivors.
    return values[:k]


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
