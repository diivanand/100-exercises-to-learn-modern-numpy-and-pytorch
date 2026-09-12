# =============================================================================
#  11.04 -- Scaled dot-product attention
# =============================================================================
#
#  Attention is three matrix products and a softmax:
#
#      scores  = Q Kᵀ / sqrt(d)          (..., T, T)
#      weights = softmax(scores, dim=-1)  each query row sums to one
#      output  = weights V                (..., T, d)
#
#  Godoy derives it over ch. 9 ("Scaled Dot Product", Equation 9.6) and the
#  reason for the divisor is in the name: the dot product of two vectors with
#  unit-variance entries has variance d, so for d = 256 the logits have a
#  standard deviation of 16, the softmax becomes a hard argmax, and the
#  gradient through every key but one is zero. Dividing by sqrt(d) puts the
#  variance back to 1. Every attention paper since Vaswani et al. (2017)
#  includes it, and every second hand-written implementation forgets it.
#
#  Causal (decoder) attention masks the future: position t may look at keys
#  0..t only. The mask is applied to the SCORES with -inf before the softmax,
#  so the masked positions get probability exactly 0 and the row still sums
#  to one. Masking after the softmax, or with a large negative number added
#  after, breaks that.
#
#  Since PyTorch 2.0 you should not write those three lines yourself in
#  production code:
#
#      F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
#
#  chooses a fused kernel (FlashAttention-2 or memory-efficient attention on
#  the 4090) that never materialises the (T, T) score matrix, which is what
#  makes long contexts fit in memory. It takes (..., T, d) for q, k and v,
#  with a heads dimension anywhere before T, and applies the scaling itself.
#  Writing the reference version by hand is still worth doing once, and that
#  is what the tests check: the two must agree.
#
#  TASK
#    Fix `attention_by_hand` (the scaling and the mask), and make `attention`
#    call the fused kernel with the causal flag.
#
#  RUN IT
#    ./npt test 11_04
#
# =============================================================================

import math  # noqa: F401

import torch
import torch.nn.functional as F


def attention_by_hand(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Reference attention written out in full. Shapes are (..., T, d)."""
    d = q.shape[-1]
    # TODO: the scores are not scaled by 1/sqrt(d).
    scores = q @ k.transpose(-2, -1)
    weights = torch.softmax(scores, dim=-1)
    if causal:
        t = q.shape[-2]
        # TODO: masking after the softmax leaves rows that no longer sum to
        # one. Mask the scores with -inf before the softmax instead.
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        weights = weights.masked_fill(~mask, 0.0)
    return weights @ v


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """The same computation through the fused kernel."""
    # TODO: pass the causal flag through as `is_causal`.
    return F.scaled_dot_product_attention(q, k, v)


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
