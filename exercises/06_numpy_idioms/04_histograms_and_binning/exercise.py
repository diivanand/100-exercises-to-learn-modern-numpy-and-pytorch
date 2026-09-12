# =============================================================================
#  06.04 -- Histograms and binning
# =============================================================================
#
#  Three functions, three conventions, and the off-by-ones live where they
#  meet.
#
#  `np.histogram(x, bins=edges)` counts values into len(edges) - 1 bins.
#  Every bin is half-open, [e_i, e_{i+1}), EXCEPT the last, which is closed
#  so that the maximum value is not thrown away. Johansson, *Numerical
#  Python* ch. 12, uses it with `density=True` for normalised histograms.
#
#  `np.digitize(x, edges)` answers "which bin?" per value, and its answer
#  is 1-BASED: it returns i such that edges[i-1] <= x < edges[i], so 0 means
#  "below the first edge" and len(edges) means "at or above the last". If
#  you want a 0-based bin index for values you already know are in range,
#  the tidy trick is to search only the INTERIOR edges, `edges[1:-1]`:
#  the result is then 0 .. n_bins - 1, and a value equal to the last edge
#  lands in the last bin, matching np.histogram.
#
#  `np.bincount(idx, weights=..., minlength=n)` is the histogram of integer
#  indices: O(n), no edges to search, and the way to sum a weight per bin
#  (per class, per bucket, per anything you can number). Always pass
#  minlength, or the output shrinks whenever the top bins are empty.
#
#  TASK
#    Make `assign_bins` 0-based and consistent with the histogram, and
#    implement the other two.
#
#  RUN IT
#    ./npt test 06_04
#
# =============================================================================

import numpy as np


def histogram_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """How many values fall in each of the len(edges) - 1 bins [e_i, e_{i+1})."""
    # TODO: np.histogram returns (counts, edges).
    return np.zeros(len(edges) - 1, dtype=int)


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The 0-based bin index of every value of `x` (all within [edges[0], edges[-1]])."""
    # TODO: digitize is 1-based, and puts a value equal to the last edge
    # into a bin that does not exist.
    return np.digitize(x, edges)


def weighted_counts(bins: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Sum of `weights` per bin index, with exactly `n_bins` entries."""
    # TODO: np.bincount with weights and minlength.
    return np.array([weights[bins == b].sum() for b in range(bins.max() + 1)])


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
