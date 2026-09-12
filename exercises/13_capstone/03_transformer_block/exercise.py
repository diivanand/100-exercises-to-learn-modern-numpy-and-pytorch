# =============================================================================
#  13.03 -- Capstone: the transformer block
# =============================================================================
#
#  A transformer is a stack of identical blocks, and a block is two
#  sub-layers: the causal self-attention from 13.02, which moves information
#  BETWEEN positions, and a small MLP applied to every position on its own,
#  which computes WITH it. Volpe et al. ch. 8, "Understanding the Transformer
#  Structure" and "Implementing a Transformer Encoder Layer"; Godoy ch. 10,
#  "Transform and Roll Out".
#
#  Two details decide whether a stack of these trains:
#
#   * RESIDUAL CONNECTIONS. Each sub-layer ADDS to its input rather than
#     replacing it: x = x + sublayer(x). The sum is the "residual stream", and
#     it gives the gradient an identity path from the loss straight back to
#     the embeddings, however many blocks are in between. A sub-layer that is
#     not yet useful can output roughly zero and do no harm.
#
#   * WHERE THE LAYERNORM GOES. LayerNorm (Godoy ch. 10, "Layer
#     Normalization") rescales each position's vector to zero mean and unit
#     variance. The 2017 paper put it AFTER the residual add ("post-LN"):
#     x = LN(x + sublayer(x)). Nearly every model since GPT-2 puts it BEFORE
#     the sub-layer, on the copy the sub-layer reads ("pre-LN" or norm-first;
#     `nn.TransformerEncoderLayer(norm_first=True)`, as Godoy notes):
#
#         x = x + attn(LN1(x))
#         x = x + mlp(LN2(x))
#
#     Pre-LN keeps the residual stream un-normalised, so the identity path is
#     really an identity, and deep stacks train without learning-rate warm-up
#     heroics. The tests pin this down structurally: with both sub-layers
#     zeroed, a pre-LN block IS the identity; a post-LN block is LN(x).
#
#  The MLP is Linear(d, 4d) -> GELU -> Linear(4d, d). The factor of four is
#  conventional (it is where most of the parameters live), and GELU rather
#  than ReLU is what the language models use (09.09).
#
#  TASK
#    Rewrite `Block.forward` as pre-LN with a residual around each sub-layer.
#
#  RUN IT
#    ./npt test 13_03
#
# =============================================================================

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
        # TODO: this is the 2017 post-LN layout, and the MLP has lost its
        # residual connection altogether. Make it pre-LN: each sub-layer reads
        # a normalised COPY of the stream and adds its result back to the
        # un-normalised stream.
        x = self.ln1(x + self.attn(x))
        x = self.ln2(self.mlp(x))
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
