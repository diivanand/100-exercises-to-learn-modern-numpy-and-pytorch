# =============================================================================
#  12.03 -- torch.testing.assert_close
# =============================================================================
#
#  Two correct implementations of the same function do not produce equal
#  floats. They add in a different order, one uses a fused multiply-add,
#  one runs on a Tensor Core with TF32 (11.06). `torch.equal(a, b)` is
#  therefore the wrong test almost every time, and `assert (a == b).all()` is
#  the same mistake with more typing. Effective Python, 3rd ed., Item 113
#  makes the point for plain floats with `assertAlmostEqual`; for tensors
#  the tool is
#
#      torch.testing.assert_close(actual, expected, rtol=..., atol=...)
#
#  It passes when |actual - expected| <= atol + rtol * |expected| holds
#  elementwise, with defaults chosen per dtype (1.3e-6 / 1e-5 for float32,
#  1.6e-2 / 1e-5 for bfloat16). When it fails it tells you HOW MANY elements
#  mismatched, the largest absolute and relative difference, and where. It
#  also checks that shape, dtype and device agree, because a (3,) that
#  happens to broadcast against a (1, 3) is a bug you want to hear about,
#  and `equal_nan=True` lets nan match nan at the same positions.
#
#  The exercise is the tests themselves: five things a good numerical test
#  checks, written against a hand-rolled LayerNorm. The last three are the
#  ones people leave out -- a loose tolerance that is loose ON PURPOSE and
#  says so, a dtype mismatch that must not be papered over, and nan
#  handling.
#
#  TASK
#    Replace the exact comparisons with assert_close, using explicit
#    tolerances where the comparison needs them.
#
#  RUN IT
#    ./npt test 12_03
#
# =============================================================================

import torch


def layer_norm_by_hand(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise the last axis to zero mean and unit variance."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def layer_norm_library(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps)


def test_hand_and_library_layer_norm_agree():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 3 + 1
    # TODO: two correct implementations differ in the last bits. Compare
    # with torch.testing.assert_close and its default tolerances.
    assert torch.equal(layer_norm_by_hand(x), layer_norm_library(x))


def test_a_small_wrong_eps_is_caught():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 1e-3  # small variance: eps matters here
    wrong = layer_norm_by_hand(x, eps=1e-3)
    right = layer_norm_library(x, eps=1e-5)
    try:
        torch.testing.assert_close(wrong, right)
    except AssertionError as e:
        assert "Mismatched elements" in str(e)
    else:
        raise AssertionError("assert_close accepted a wrong eps")


def test_tolerances_are_explicit_when_they_must_be_loose():
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    half = layer_norm_by_hand(x.to(torch.bfloat16)).to(torch.float32)
    full = layer_norm_by_hand(x)
    # TODO: bfloat16 has 8 bits of mantissa, ~3 significant digits. The
    # default float32 tolerance cannot pass; say what tolerance you mean
    # (rtol and atol of about 2e-2) rather than dropping the check.
    torch.testing.assert_close(half, full)


def test_dtype_and_shape_mismatches_are_errors_not_near_misses():
    a = torch.zeros(3)
    try:
        torch.testing.assert_close(a, a.to(torch.float64))
    except AssertionError as e:
        assert "dtype" in str(e)
    else:
        raise AssertionError("a dtype mismatch was accepted")
    try:
        torch.testing.assert_close(a, torch.zeros(1, 3))
    except AssertionError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("a shape mismatch was accepted")


def test_nan_positions_must_agree_when_asked():
    a = torch.tensor([1.0, float("nan")])
    # TODO: nan != nan, so this comparison can never pass. assert_close with
    # equal_nan=True treats nans at the same positions as equal.
    assert torch.equal(a, a.clone())
    try:
        torch.testing.assert_close(a, torch.tensor([1.0, 2.0]), equal_nan=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("nan matched a number")
