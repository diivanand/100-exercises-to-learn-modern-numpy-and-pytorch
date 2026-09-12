# =============================================================================
#  01.02 -- Dtypes and overflow
# =============================================================================
#
#  Python integers have no limit. NumPy integers are C integers: fixed width,
#  and arithmetic that leaves the range WRAPS. `np.uint8(250) + 10` is not
#  260; the array version gives 4, silently. Johansson, ch. 2, "Data Types"
#  (Table 2-2) lists the widths; the rest of this exercise is about what the
#  widths cost you.
#
#  The classic case is image data. Pixels are uint8 because that is what
#  files and screens use, and it is one byte per pixel instead of eight. The
#  moment you do arithmetic on them you have to decide where the arithmetic
#  happens:
#
#      image + 50                     # uint8 + Python int -> uint8, wraps
#      image.astype(np.int16) + 50    # room for the true result
#      np.clip(..., 0, 255).astype(np.uint8)   # and back, deliberately
#
#  Under NumPy 2 (NEP 50) a Python scalar is "weak": it adopts the array's
#  dtype rather than widening it. That is a good rule -- `image * 0.5` should
#  not turn a float32 image into float64 -- but it means the width of the
#  RESULT is decided by your array, and a Python int that does not fit that
#  width at all (`image + 300`) is an OverflowError rather than a wrap.
#
#  `astype` is the other place values get lost. Casting 200 to int8 gives
#  -56 with no warning: astype is a reinterpretation, not a conversion with
#  range checks. If you mean "clamp to what fits", write the clamp:
#  `np.clip(values, np.iinfo(np.int8).min, np.iinfo(np.int8).max)`.
#
#  Floats lose precision rather than wrapping. float32 has 24 significant
#  bits, so it can count exactly to 16,777,216 and not one further; float64
#  has 53. Half the memory, half the bandwidth, and enough for most machine
#  learning, but not for accumulating a million values in a running sum.
#
#  TASK
#    Make `brighten` saturate at 255, make `to_int8_saturating` clamp instead
#    of wrap, and write `as_float32`.
#
#  RUN IT
#    ./npt test 01_02
#
# =============================================================================

import numpy as np


def brighten(image: np.ndarray, amount: int) -> np.ndarray:
    """Add `amount` to every pixel of a uint8 image, saturating at 255."""
    # TODO: uint8 + int stays uint8 and wraps at 256. Widen first, clip, then
    # narrow back on purpose.
    return image + amount


def to_int8_saturating(values: np.ndarray) -> np.ndarray:
    """Convert to int8, clamping anything outside [-128, 127] to the nearest edge."""
    # TODO: astype does not clamp; it keeps the low 8 bits. Clip first, and
    # get the limits from np.iinfo rather than typing them.
    return values.astype(np.int8)


def as_float32(values: np.ndarray) -> np.ndarray:
    """A float32 copy of `values`, whatever its dtype."""
    # TODO
    raise NotImplementedError


def test_brighten_saturates_instead_of_wrapping():
    image = np.array([[0, 100], [200, 250]], dtype=np.uint8)
    bright = brighten(image, 50)
    assert bright.dtype == np.uint8
    np.testing.assert_array_equal(bright, [[50, 150], [250, 255]])


def test_brighten_leaves_the_input_alone():
    image = np.array([10, 20], dtype=np.uint8)
    brighten(image, 5)
    np.testing.assert_array_equal(image, [10, 20])


def test_to_int8_saturating():
    values = np.array([-1000, -128, 0, 127, 200, 1000])
    result = to_int8_saturating(values)
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result, [-128, -128, 0, 127, 127, 127])


def test_as_float32_halves_the_memory():
    doubles = np.arange(1000, dtype=np.float64)
    singles = as_float32(doubles)
    assert singles.dtype == np.float32
    assert singles.nbytes == doubles.nbytes // 2
    # And is not exact: float32 has 24 bits of significand.
    assert as_float32(np.array([16_777_217.0]))[0] == 16_777_216.0
