# Solution -- 12.03 torch.testing.assert_close

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
    # Two correct implementations differ in the last bits: they sum in a
    # different order. Compare with a tolerance, never with ==.
    torch.testing.assert_close(layer_norm_by_hand(x), layer_norm_library(x))


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
    # bfloat16 has 8 bits of mantissa: ~3 significant decimal digits.
    torch.testing.assert_close(half, full, rtol=2e-2, atol=2e-2)


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
    torch.testing.assert_close(a, a.clone(), equal_nan=True)
    try:
        torch.testing.assert_close(a, torch.tensor([1.0, 2.0]), equal_nan=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("nan matched a number")
