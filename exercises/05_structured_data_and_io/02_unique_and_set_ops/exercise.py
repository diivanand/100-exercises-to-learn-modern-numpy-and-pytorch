# =============================================================================
#  05.02 -- unique and set operations
# =============================================================================
#
#  `np.unique` sorts and de-duplicates, and with the right keywords it hands
#  back the two things you usually wanted alongside: how often each value
#  occurs (`return_counts=True`) and, for every original element, the index
#  of its value in the de-duplicated array (`return_inverse=True`). That
#  inverse is an integer encoding of a categorical column, and
#  `unique[inverse]` reconstructs the original. NumPy 2.0 added one function
#  per combination -- `np.unique_counts`, `np.unique_inverse`,
#  `np.unique_values`, `np.unique_all` -- returning named tuples, which read
#  better than remembering the order of a four-tuple.
#
#  Membership is `np.isin(needles, haystack)`: a boolean mask with the shape
#  of the FIRST argument. Johansson, *Numerical Python* ch. 2, "Set
#  Operations", shows the same thing spelt `np.in1d`; that name was removed in
#  NumPy 2.0 (it only handled 1-D arrays, hence the name), so treat any code
#  you find using it as a migration job. `np.union1d`, `np.intersect1d` and
#  `np.setdiff1d` are the other set operations, all returning sorted unique
#  values.
#
#  TASK
#    Implement the three functions with unique/isin, without Python loops.
#
#  RUN IT
#    ./npt test 05_02
#
# =============================================================================

import numpy as np


def label_frequencies(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct labels, sorted, and how often each occurs."""
    # TODO: one call to np.unique_counts (or np.unique with return_counts).
    values = np.unique(labels)
    counts = np.array([(labels == v).sum() for v in values])
    return values, counts


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes for each label (index into the sorted vocabulary) and
    the vocabulary itself, so that vocabulary[codes] reconstructs `labels`."""
    # TODO: codes must index into the SORTED vocabulary. This assigns codes
    # in order of first appearance, so vocabulary[codes] is not `labels`.
    seen: dict[str, int] = {}
    codes = np.array([seen.setdefault(str(x), len(seen)) for x in labels])
    return codes, np.unique(labels)


def unseen(values: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """The entries of `values`, in order, that do not appear in `vocabulary`."""
    # TODO: the arguments to np.isin are the wrong way round.
    return values[~np.isin(vocabulary, values)]


def test_label_frequencies():
    labels = np.array(["cat", "dog", "cat", "bird", "cat", "dog"])
    values, counts = label_frequencies(labels)
    assert list(values) == ["bird", "cat", "dog"]
    assert list(counts) == [1, 3, 2]


def test_encode_labels_round_trips():
    labels = np.array(["red", "green", "red", "blue", "green"])
    codes, vocabulary = encode_labels(labels)
    assert list(vocabulary) == ["blue", "green", "red"]
    assert list(codes) == [2, 1, 2, 0, 1]
    np.testing.assert_array_equal(vocabulary[codes], labels)
    assert codes.dtype.kind == "i"


def test_unseen_keeps_order_and_duplicates():
    values = np.array([5, 1, 9, 1, 7, 3])
    vocabulary = np.array([1, 3, 4])
    np.testing.assert_array_equal(unseen(values, vocabulary), [5, 9, 7])


def test_unseen_shapes_follow_the_first_argument():
    # Argument order matters: isin(a, b) is a mask over a. Swapped, the
    # result has the wrong length and indexing raises or answers nonsense.
    values = np.arange(10)
    vocabulary = np.array([2, 4])
    assert unseen(values, vocabulary).shape == (8,)
    assert unseen(vocabulary, values).shape == (0,)
