# =============================================================================
#  13.02 -- Capstone: causal self-attention
# =============================================================================
#
#  Attention lets every position build its output from a weighted mixture of
#  every other position, with weights the model computes from the data. For
#  each position a QUERY vector is compared (dot product) with every KEY
#  vector; the scores are scaled by 1/sqrt(head_dim), softmaxed, and used to
#  average the VALUE vectors. Volpe et al. ch. 8, "Implementing Dot-Product
#  Attention", and Godoy ch. 9, "Self-Attention", build this up by hand; you
#  did the same in 11.04.
#
#  Three things turn that formula into the block a language model uses:
#
#   * SELF-attention: q, k and v all come from the same input, through one
#     Linear(d, 3d) that is then split three ways.
#
#   * MULTI-HEAD: the d_model features are cut into n_heads slices, and
#     attention runs independently in each slice (Volpe et al. ch. 8,
#     "Breaking Down Multi-Head Attention"). The reshape is the classic
#     source of silent bugs: (B, T, D) -> (B, T, H, hd) -> transpose(1, 2) ->
#     (B, H, T, hd). Attention then runs over the last two dims, time and
#     head_dim, for every (batch, head). Skip the transpose and you have carved
#     heads out of the TIME axis: the shapes still work, the numbers are
#     nonsense.
#
#   * CAUSAL: position t may only see positions 0..t, because at generation
#     time the future does not exist yet. The mask sets the scores of future
#     positions to -inf BEFORE the softmax, so their weight is exactly zero
#     (Volpe et al. ch. 8, "Adding Masking to the Multi-Head Attention
#     Layer"). An off-by-one in the mask lets every position peek one token
#     ahead: training loss looks wonderful, generation is garbage.
#
#  `F.scaled_dot_product_attention(q, k, v, is_causal=True)` does the scaling,
#  masking, softmax and averaging in one fused call, picking a memory-
#  efficient kernel where it can (11.04). Prefer it to the written-out formula;
#  the formula is kept here, in `reference_attention`, as the specification
#  the tests compare against.
#
#  TASK
#    Fix the head reshape, and replace the hand-built mask with
#    `is_causal=True` (or fix the mask -- the test does not care which).
#
#  RUN IT
#    ./npt test 13_02
#
# =============================================================================

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t sees positions 0..t only."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # One matrix produces q, k and v for every head at once: three
        # separate Linear layers would do three smaller matmuls for the same
        # arithmetic.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=-1)
        # TODO: this reshapes (B, T, D) straight to (B, H, T, hd), which slices
        # the heads out of the TIME axis. Reshape to (B, T, H, hd) first, then
        # transpose(1, 2), so each head gets a slice of the FEATURES at every
        # time step.
        q = q.view(batch, self.n_heads, length, self.head_dim)
        k = k.view(batch, self.n_heads, length, self.head_dim)
        v = v.view(batch, self.n_heads, length, self.head_dim)
        # TODO: this mask is off by one: diagonal=1 keeps the first
        # super-diagonal, so position t can attend to t + 1. The fused call has
        # `is_causal=True` for exactly this mask; use it and drop `attn_mask`.
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # (B, H, T, hd) -> (B, T, H, hd) -> (B, T, D). contiguous() is needed
        # because view() cannot merge dims that a transpose left apart (07.04).
        out = out.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.proj(out)


def reference_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """The textbook formula, written out, for (..., T, hd) tensors."""
    length = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_output_shape():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == (2, 10, 32)


def test_matches_the_written_out_formula():
    torch.manual_seed(1)
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(3, 7, 32)
    got = attn(x)

    # Recompute with the module's own weights and the explicit formula.
    batch, length, d_model = x.shape
    q, k, v = attn.qkv(x).split(d_model, dim=-1)
    split = lambda t: t.view(batch, length, 4, 8).transpose(1, 2)  # noqa: E731
    ref = reference_attention(split(q), split(k), split(v))
    ref = attn.proj(ref.transpose(1, 2).reshape(batch, length, d_model))
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_the_future_does_not_leak_into_the_past():
    torch.manual_seed(2)
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(1, 12, 32)
    y = attn(x)

    # Change the input from position 6 on. Outputs at positions 0..5 must be
    # bit-for-bit unchanged; an off-by-one mask would let position 5 move.
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 6, 32)
    y2 = attn(x2)
    torch.testing.assert_close(y2[:, :6], y[:, :6])
    assert not torch.allclose(y2[:, 6:], y[:, 6:])


def test_heads_are_split_over_features_not_time():
    torch.manual_seed(3)
    attn = CausalSelfAttention(d_model=8, n_heads=2).eval()
    # A sequence of one position, then the same position repeated: if heads
    # were carved out of the TIME axis, the second call would see a different
    # per-head layout and position 0 would change.
    x = torch.randn(1, 1, 8)
    single = attn(x)
    repeated = attn(x.repeat(1, 4, 1))
    torch.testing.assert_close(repeated[:, 0], single[:, 0])


def test_rejects_indivisible_head_count():
    import pytest

    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=30, n_heads=4)
