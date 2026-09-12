# =============================================================================
#  05.05 -- datetime64 and strings
# =============================================================================
#
#  NumPy has a date type, `datetime64`, and it carries a UNIT: "D" for days,
#  "s" for seconds, "ns" for nanoseconds. Subtracting two datetime64 arrays
#  gives a `timedelta64` in the finer of the two units, and that is the trap:
#  a day-precision start and a nanosecond-precision end give a difference
#  in nanoseconds, so `.astype(int)` yields 259_200_000_000_000 where you
#  expected 3. Convert the unit explicitly, `.astype("timedelta64[D]")`, or
#  divide by `np.timedelta64(1, "D")`, before turning it into a number.
#  Casting a date DOWN to a coarser unit truncates, which is the trick for
#  "which month is this": `d.astype("datetime64[M]")` is the first of the
#  month, and subtracting `d.astype("datetime64[Y]")` counts months since
#  January. (Effective Python Item 105 covers Python's datetime for local
#  clocks; datetime64 is UTC-only and has no time zones by design.)
#
#  Strings in NumPy used to mean fixed-width dtypes ("U10") and a `np.char`
#  module that looped in Python behind the scenes. NumPy 2.0 added
#  `np.strings`, real ufuncs over string arrays (`np.strings.upper`,
#  `.strip`, `.find`, `.replace`, ...), and a variable-width `StringDType`
#  (`np.dtypes.StringDType()`) that stores each string separately, like a
#  Python list, without the padding. Use np.strings for anything that would
#  otherwise be a loop over `str` methods.
#
#  TASK
#    Get the units right in `days_between`, compute `month_of` with unit
#    casts, and rewrite `normalise_names` on np.strings.
#
#  RUN IT
#    ./npt test 05_05
#
# =============================================================================

import numpy as np


def days_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Whole days from each `start` to each `end` (both datetime64 arrays)."""
    # TODO: the difference is in whatever unit the finer input has. Cast it
    # to days before converting to integers.
    return (end - start).astype(np.int64)


def month_of(dates: np.ndarray) -> np.ndarray:
    """The month number (1-12) of each date."""
    # TODO: no loop over Python datetime objects. Cast to "datetime64[M]" and
    # "datetime64[Y]" and subtract.
    return np.array([int(str(d)[5:7]) for d in dates])


def normalise_names(names: np.ndarray) -> np.ndarray:
    """Strip surrounding whitespace and lower-case every name."""
    # TODO: np.strings.strip and np.strings.lower, not a Python loop.
    return np.array([str(n).strip().lower() for n in names])


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
