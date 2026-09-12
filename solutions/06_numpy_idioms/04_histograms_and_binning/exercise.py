# Solution -- 06.04 Histograms and binning

import numpy as np


def histogram_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """How many values fall in each of the len(edges) - 1 bins [e_i, e_{i+1})."""
    # np.histogram with explicit edges. Its last bin is closed on the right,
    # so a value equal to edges[-1] is counted, not dropped.
    counts, _ = np.histogram(x, bins=edges)
    return counts


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The 0-based bin index of every value of `x` (all within [edges[0], edges[-1]])."""
    # np.digitize returns i such that edges[i-1] <= x < edges[i]: 1-based,
    # with 0 and len(edges) for values outside. Dropping the outer edges
    # from the search shifts everything to 0-based and maps the right-hand
    # end point into the last bin, matching np.histogram's convention.
    return np.digitize(x, edges[1:-1])


def weighted_counts(bins: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Sum of `weights` per bin index, with exactly `n_bins` entries."""
    # bincount is the integer-index histogram: O(n), no edges to search.
    # minlength keeps the output length fixed even if the last bins are empty.
    return np.bincount(bins, weights=weights, minlength=n_bins)


def test_histogram_counts():
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(histogram_counts(x, edges), [2, 2, 3])


def test_assign_bins_is_zero_based_and_matches_histogram():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 10.0, size=1000)
    edges = np.linspace(0.0, 10.0, 6)
    bins = assign_bins(x, edges)
    assert bins.min() >= 0 and bins.max() <= 4
    np.testing.assert_array_equal(
        np.bincount(bins, minlength=5), histogram_counts(x, edges)
    )
    # A value on an interior edge belongs to the bin on its right; the very
    # last edge belongs to the last bin.
    np.testing.assert_array_equal(
        assign_bins(np.array([0.0, 2.0, 10.0]), edges), [0, 1, 4]
    )


def test_weighted_counts():
    bins = np.array([0, 2, 2, 1, 0])
    weights = np.array([1.0, 10.0, 100.0, 5.0, 2.0])
    np.testing.assert_array_equal(
        weighted_counts(bins, weights, 4), [3.0, 5.0, 110.0, 0.0]
    )
