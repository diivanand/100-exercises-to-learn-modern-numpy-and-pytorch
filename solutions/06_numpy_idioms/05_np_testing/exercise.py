# Solution -- 06.05 np.testing

import numpy as np


def assert_matrices_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Raise AssertionError unless `actual` matches `expected` to floating-point
    tolerance, treating NaN at the same positions as equal."""
    # assert_allclose: |actual - expected| <= atol + rtol * |expected|.
    # rtol handles the size of the numbers; atol handles values near zero,
    # where any relative tolerance collapses to "must be exactly zero".
    # equal_nan=True is the default here, and it is spelt out because the
    # exercise is about knowing that.
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12, equal_nan=True)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """||actual - expected|| / ||expected||, the number to report in a bug."""
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


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
