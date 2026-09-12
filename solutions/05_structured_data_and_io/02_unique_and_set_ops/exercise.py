# Solution -- 05.02 unique and set operations

import numpy as np


def label_frequencies(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct labels, sorted, and how often each occurs."""
    # np.unique_counts (NumPy 2.0) returns a small named tuple. The older
    # spelling np.unique(labels, return_counts=True) is equivalent.
    result = np.unique_counts(labels)
    return result.values, result.counts


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes for each label (index into the sorted vocabulary) and
    the vocabulary itself, so that vocabulary[codes] reconstructs `labels`."""
    result = np.unique_inverse(labels)
    return result.inverse_indices, result.values


def unseen(values: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """The entries of `values`, in order, that do not appear in `vocabulary`."""
    # np.isin(needles, haystack): the mask has the shape of the FIRST
    # argument. It replaced np.in1d, which NumPy 2.0 removed.
    return values[~np.isin(values, vocabulary)]


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
