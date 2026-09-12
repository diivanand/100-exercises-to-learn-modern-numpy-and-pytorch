# =============================================================================
#  06.05 -- np.testing
# =============================================================================
#
#  0.1 + 0.2 == 0.3 is False, and it always will be. Effective Python
#  (3rd ed.), Item 113, makes the case for `assertAlmostEqual` in unittest;
#  the NumPy equivalents live in `np.testing` and this course has used them
#  from the first chapter. It is time to look at what they do.
#
#      np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)
#
#  passes when |actual - expected| <= atol + rtol * |expected| everywhere.
#  `rtol` scales with the magnitude of the numbers, which is what rounding
#  error does; `atol` matters near zero, where a purely relative test
#  demands exact equality. Choose both on purpose: rtol=1e-7 is about the
#  precision of float32 and a comfortable margin for float64 arithmetic;
#  atol=1e-12 forgives a value that "should be" zero and came out as 3e-17.
#  Set `equal_nan=True` (the default in assert_allclose, but not in
#  np.isclose) when NaN at the same position is the expected answer:
#  nan != nan, so `==` can never confirm it. The failure message lists the
#  worst mismatches, which is more useful than a bare False.
#
#  `assert_array_equal` is for integers, booleans and strings, where
#  equality is exact; `assert_array_almost_equal` is the old decimal-places
#  interface, superseded by assert_allclose.
#
#  TASK
#    Rewrite `assert_matrices_close` on assert_allclose with the tolerances
#    above, and implement `relative_error`.
#
#  RUN IT
#    ./npt test 06_05
#
# =============================================================================

import numpy as np


def assert_matrices_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Raise AssertionError unless `actual` matches `expected` to floating-point
    tolerance, treating NaN at the same positions as equal."""
    # TODO: exact comparison rejects round-off and says NaN != NaN.
    if not np.array_equal(actual, expected):
        raise AssertionError("arrays differ")


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """||actual - expected|| / ||expected||, the number to report in a bug."""
    # TODO: np.linalg.norm of the difference over the norm of the expected.
    return 0.0


def test_accepts_round_off_differences():
    a = np.array([0.1 + 0.2, 1.0 / 3.0 * 3.0, 1e-17])
    b = np.array([0.3, 1.0, 0.0])
    assert not np.array_equal(a, b)  # `==` says no
    assert_matrices_close(a, b)  # tolerance says yes


def test_accepts_relative_differences_on_large_values():
    a = np.array([1e12, 1e12])
    b = a * (1.0 + 1e-9)
    assert_matrices_close(a, b)


def test_rejects_real_differences():
    for wrong in (np.array([1.0, 2.0, 3.5]), np.array([1.0, 2.0, 3.0 + 1e-3])):
        try:
            assert_matrices_close(wrong, np.array([1.0, 2.0, 3.0]))
        except AssertionError:
            continue
        raise AssertionError(f"{wrong} should not have been accepted")


def test_nan_in_the_same_place_is_equal():
    a = np.array([1.0, np.nan, 3.0])
    assert not np.array_equal(a, a)  # nan != nan, so even `a == a` is not all True
    assert_matrices_close(a, a.copy())


def test_relative_error():
    e = np.array([3.0, 4.0])
    assert relative_error(e, e) == 0.0
    np.testing.assert_allclose(relative_error(np.array([3.0, 4.5]), e), 0.1)
