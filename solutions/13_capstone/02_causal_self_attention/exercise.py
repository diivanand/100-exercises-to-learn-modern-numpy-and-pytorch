# Solution -- 13.02 Capstone: causal self-attention

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
        # (B, T, D) -> (B, T, H, hd) -> (B, H, T, hd). The transpose matters:
        # attention is computed per head over TIME, so time and head must be
        # the last two dims before the head dim, not interleaved.
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
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
