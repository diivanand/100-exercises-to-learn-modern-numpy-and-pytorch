# Solution -- 01.02 Dtypes and overflow
import numpy as np


def brighten(image: np.ndarray, amount: int) -> np.ndarray:
    """Add `amount` to every pixel of a uint8 image, saturating at 255."""
    # Do the arithmetic in a type wide enough to hold the true result, clip
    # to the representable range, and only then go back to uint8. The cast at
    # the end is a deliberate, visible narrowing (compare C++ braced init).
    wide = image.astype(np.int16) + amount
    return np.clip(wide, 0, 255).astype(np.uint8)


def to_int8_saturating(values: np.ndarray) -> np.ndarray:
    """Convert to int8, clamping anything outside [-128, 127] to the nearest edge."""
    info = np.iinfo(np.int8)
    return np.clip(values, info.min, info.max).astype(np.int8)


def as_float32(values: np.ndarray) -> np.ndarray:
    """A float32 copy of `values`, whatever its dtype."""
    return values.astype(np.float32)


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
