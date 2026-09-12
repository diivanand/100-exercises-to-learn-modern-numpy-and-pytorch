# Solution -- 11.04 Scaled dot-product attention

import math

import torch
import torch.nn.functional as F


def attention_by_hand(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Reference attention written out in full. Shapes are (..., T, d)."""
    d = q.shape[-1]
    # Without the 1/sqrt(d) the logits grow with d and the softmax saturates:
    # almost all the weight lands on one key, and the gradient through the
    # other keys is ~0 (Godoy ch. 9, "Scaled Dot Product").
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        t = q.shape[-2]
        # Masking must happen BEFORE the softmax, with -inf, so that the
        # masked keys contribute exactly zero probability and the rows still
        # sum to one.
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """The same computation through the fused kernel."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def make_qkv(batch: int = 2, heads: int = 4, t: int = 6, d: int = 8):
    torch.manual_seed(0)
    return (torch.randn(batch, heads, t, d) for _ in range(3))


def test_by_hand_agrees_with_the_fused_kernel():
    q, k, v = make_qkv()
    torch.testing.assert_close(attention_by_hand(q, k, v), attention(q, k, v))


def test_causal_masking_agrees_too():
    q, k, v = make_qkv()
    torch.testing.assert_close(
        attention_by_hand(q, k, v, causal=True), attention(q, k, v, causal=True)
    )


def test_causal_output_ignores_the_future():
    q, k, v = make_qkv()
    out = attention_by_hand(q, k, v, causal=True)
    # Change the last key/value; positions before it must be unaffected.
    k2, v2 = k.clone(), v.clone()
    k2[..., -1, :] += 10.0
    v2[..., -1, :] += 10.0
    out2 = attention_by_hand(q, k2, v2, causal=True)
    torch.testing.assert_close(out[..., :-1, :], out2[..., :-1, :])
    assert not torch.allclose(out[..., -1, :], out2[..., -1, :])


def test_scaling_keeps_the_softmax_from_saturating():
    # With d = 256 and unit-variance inputs, unscaled logits have std 16 and
    # the softmax puts ~all its mass on one key. Scaled logits have std 1.
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, 16, 256) for _ in range(3))
    out = attention_by_hand(q, k, v)
    # The output should be a genuine mixture of several values, so its norm
    # is well below the norm of any single value row (about sqrt(256) = 16).
    assert out.norm(dim=-1).mean() < 8.0
