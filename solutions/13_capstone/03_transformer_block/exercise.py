# Solution -- 13.03 Capstone: the transformer block

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --- from 13.02, unchanged --------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.proj(out)


# --- this exercise ------------------------------------------------------------
class MLP(nn.Module):
    """Position-wise feed-forward network: widen, GELU, narrow."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Norm-first: the residual stream `x` is never normalised itself, only
        # the copy each sub-layer reads. That keeps an identity path from
        # input to output through every block, which is why deep stacks of
        # these train without warm-up tricks (Godoy ch. 10 contrasts norm-last
        # and norm-first; nn.TransformerEncoderLayer(norm_first=True) is this).
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def test_block_preserves_shape():
    torch.manual_seed(0)
    block = Block(d_model=32, n_heads=4)
    x = torch.randn(2, 9, 32)
    assert block(x).shape == (2, 9, 32)


def test_block_is_causal():
    torch.manual_seed(1)
    block = Block(d_model=32, n_heads=4).eval()
    x = torch.randn(1, 12, 32)
    y = block(x)
    x2 = x.clone()
    x2[:, 8:] = torch.randn(1, 4, 32)
    y2 = block(x2)
    torch.testing.assert_close(y2[:, :8], y[:, :8])


def test_residual_stream_is_an_identity_path():
    # With both sub-layers producing zero, a pre-norm block is exactly the
    # identity. A post-norm block would return LayerNorm(x) instead, which
    # is not x: that is the structural difference this test pins down.
    torch.manual_seed(2)
    block = Block(d_model=16, n_heads=2).eval()
    with torch.no_grad():
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()
        block.mlp.proj.weight.zero_()
        block.mlp.proj.bias.zero_()
    x = torch.randn(3, 5, 16) * 10 + 3  # far from zero-mean/unit-variance
    torch.testing.assert_close(block(x), x)


def test_each_sublayer_reads_a_normalised_copy():
    # Scale the input by 1000. LayerNorm removes the scale before the
    # attention sub-layer sees it, so the attention CONTRIBUTION (output minus
    # the residual) must be unchanged. If attention read the raw stream, its
    # contribution would blow up with the input.
    torch.manual_seed(3)
    block = Block(d_model=16, n_heads=2).eval()
    with torch.no_grad():
        block.mlp.proj.weight.zero_()
        block.mlp.proj.bias.zero_()
    x = torch.randn(1, 6, 16)
    small = block(x) - x
    big = block(1000 * x) - 1000 * x
    torch.testing.assert_close(small, big, rtol=1e-3, atol=1e-4)


def test_mlp_widens_by_four_and_uses_gelu():
    mlp = MLP(d_model=8)
    assert mlp.fc.out_features == 32
    assert mlp.proj.in_features == 32
    x = torch.randn(4, 8)
    with torch.no_grad():
        mlp.proj.weight.copy_(torch.eye(8, 32))
        mlp.proj.bias.zero_()
    # With an identity projection the output is the first 8 GELU units.
    torch.testing.assert_close(mlp(x), F.gelu(mlp.fc(x))[:, :8])


def test_dropout_is_off_in_eval_mode():
    torch.manual_seed(4)
    block = Block(d_model=16, n_heads=2, dropout=0.5).eval()
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(x), block(x))
