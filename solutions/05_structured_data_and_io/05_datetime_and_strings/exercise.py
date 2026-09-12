# Solution -- 05.05 datetime64 and strings

import numpy as np


def days_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Whole days from each `start` to each `end` (both datetime64 arrays)."""
    # Subtracting two datetime64 arrays gives a timedelta64 in the finer of
    # the two units. Casting the result to "D" (or dividing by one day)
    # converts to a count of days; astype(int) then drops the unit.
    return (end - start).astype("timedelta64[D]").astype(np.int64)


def month_of(dates: np.ndarray) -> np.ndarray:
    """The month number (1-12) of each date."""
    # Truncating to month precision and subtracting the year gives a
    # timedelta in months; no Python datetime objects, no loop.
    months = dates.astype("datetime64[M]")
    years = dates.astype("datetime64[Y]")
    return (months - years).astype(np.int64) + 1


def normalise_names(names: np.ndarray) -> np.ndarray:
    """Strip surrounding whitespace and lower-case every name."""
    # np.strings (NumPy 2.0) holds ufunc-based string operations. They run
    # in C over the whole array, unlike a Python loop over str methods.
    return np.strings.lower(np.strings.strip(names))


def test_days_between():
    start = np.array(["2024-02-27", "2024-12-31"], dtype="datetime64[D]")
    end = np.array(["2024-03-01", "2025-01-01"], dtype="datetime64[D]")
    np.testing.assert_array_equal(days_between(start, end), [3, 1])  # leap year


def test_days_between_mixed_units():
    # Timestamps at nanosecond precision against dates at day precision:
    # the difference is computed in nanoseconds, and must still come back
    # as whole days, not as 259_200_000_000_000.
    start = np.array(["2024-01-01"], dtype="datetime64[D]")
    end = np.array(["2024-01-04T12:30:00.000000000"], dtype="datetime64[ns]")
    result = days_between(start, end)
    assert result.dtype.kind == "i"
    np.testing.assert_array_equal(result, [3])


def test_month_of():
    dates = np.array(["2023-01-15", "2023-07-04", "2024-12-25"], dtype="datetime64[D]")
    np.testing.assert_array_equal(month_of(dates), [1, 7, 12])


def test_normalise_names():
    names = np.array(["  Ada ", "GRACE", " linus"])
    out = normalise_names(names)
    assert list(out) == ["ada", "grace", "linus"]
    assert out.dtype.kind in "US"


def test_normalise_names_is_vectorised(monkeypatch):
    # The np.strings functions are the point of the exercise; a Python loop
    # over str.strip would pass the test above and miss it.
    called = []
    real = np.strings.strip

    def spy(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(np.strings, "strip", spy)
    normalise_names(np.array([" x "]))
    assert called, "use np.strings"
