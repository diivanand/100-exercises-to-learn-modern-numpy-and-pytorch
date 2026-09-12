# =============================================================================
#  13.04 -- Capstone: the training loop
# =============================================================================
#
#  Everything is now in place to train. The model, `TinyLM`, is embeddings,
#  a stack of blocks, a final LayerNorm and a Linear "head" that turns each
#  position's vector into scores over the vocabulary; its loss is
#  cross-entropy between those scores and the shifted targets, exactly as in
#  09.04 (logits in, no softmax). Cautaerts ch. 14, "Training the language
#  model", and Godoy ch. 2, "Rethinking the Training Loop", describe the same
#  loop; the pieces below are the ones that go wrong in practice:
#
#   * ZERO THE GRADIENTS before backward. `.grad` accumulates (08.02). Leave
#     this out and every step applies the sum of every gradient so far: the
#     loss falls for a while, then climbs, and nothing else in the code looks
#     wrong.
#
#   * CLIP THE GRADIENT NORM after backward, before the step (09.10).
#     `clip_grad_norm_(params, 1.0)` rescales the whole gradient so its total
#     norm is at most 1. Transformers occasionally produce a gradient a
#     hundred times larger than usual; clipping turns that step into a normal
#     one instead of a jump into a region the model never recovers from.
#
#   * SCHEDULE THE LEARNING RATE: a linear warm-up from ~0 to the peak over
#     the first few dozen steps (Adam's second-moment estimate is garbage at
#     step 0 and warm-up keeps the early steps small), then a cosine decay to
#     zero by the last step. The schedule is a pure function of the step,
#     written here by hand rather than through `torch.optim.lr_scheduler`, so
#     the test can check its shape point by point.
#
#   * AdamW, not Adam with L2 (09.05). Weight decay is applied to the weights
#     directly, not folded into the gradient that Adam then rescales.
#
#  The training test runs 300 steps on the 1.6 KB corpus, batch 32, context
#  64. On a laptop CPU that is about 3 seconds (measured: 2.7 s on an Apple
#  M-series). The model memorises the text: the loss falls from ln(V) = 3.6
#  to well under 1. That is not generalisation, and not meant to be; it is a
#  check that every gradient reaches every parameter.
#
#  TASK
#    Fix the schedule, zero the gradients, and clip them.
#
#  RUN IT
#    ./npt test 13_04
#
# =============================================================================

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CORPUS = """\
Alice was beginning to get very tired of sitting by her sister on the bank, and
of having nothing to do: once or twice she had peeped into the book her sister
was reading, but it had no pictures or conversations in it, 'and what is the use
of a book,' thought Alice 'without pictures or conversations?'

So she was considering in her own mind (as well as she could, for the hot day
made her feel very sleepy and stupid), whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall
be late!' (when she thought it over afterwards, it occurred to her that she
ought to have wondered at this, but at the time it all seemed quite natural);
but when the Rabbit actually took a watch out of its waistcoat-pocket, and
looked at it, and then hurried on, Alice started to her feet, for it flashed
across her mind that she had never before seen a rabbit with either a
waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she
ran across the field after it, and fortunately was just in time to see it pop
down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the
world she was to get out again.
"""


# --- from 13.01 to 13.03, unchanged -------------------------------------------
class CharTokeniser:
    def __init__(self, text: str) -> None:
        self.chars: list[str] = sorted(set(text))
        self._to_id: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> Tensor:
        return torch.tensor([self._to_id[ch] for ch in text], dtype=torch.long)

    def decode(self, ids: Tensor) -> str:
        return "".join(self.chars[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
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


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# --- this exercise ------------------------------------------------------------
class TinyLM(nn.Module):
    """A decoder-only transformer over characters."""

    def __init__(
        self,
        vocab_size: int,
        context: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: the matrix that turns a character into a vector is
        # also the one that turns a vector back into character scores. Fewer
        # parameters, and a well-known small win for language models.
        self.head.weight = self.token.weight
        # nn.Embedding initialises with N(0, 1). Through the tied head that
        # makes the initial logits about sqrt(d_model) in size and the first
        # loss around 20 instead of ln(vocab): the model starts confidently
        # wrong. GPT-2's std of 0.02 starts it near uniform.
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        _, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.token(idx) + self.position(positions)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        # cross_entropy wants (N, C) logits and (N,) class indices, so fold
        # batch and time together. It takes LOGITS: no softmax here (09.04).
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


def get_batch(
    data: Tensor, context: int, batch_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """`batch_size` random windows of the corpus, with targets shifted by one."""
    starts = torch.randint(0, len(data) - context, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + 1 : s + context + 1] for s in starts])
    return x, y


def lr_at(
    step: int, max_lr: float, warmup: int, total: int, min_lr: float = 0.0
) -> float:
    """Linear warm-up to max_lr, then a cosine decay to min_lr at `total`."""
    # TODO: this warms up over `total` steps instead of `warmup`, and never
    # decays. Warm-up: max_lr * (step + 1) / warmup while step < warmup.
    # After that, with progress = (step - warmup) / (total - warmup) clamped
    # to [0, 1]:  min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress)).
    return max_lr * (step + 1) / total


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    lr: float,
    max_norm: float = 1.0,
) -> float:
    """One optimisation step. Returns the loss as a Python float."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    # TODO: zero the gradients before backward (08.02).
    _, loss = model(x, y)
    loss.backward()
    # TODO: clip the gradient norm to `max_norm` here, between backward and
    # step: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).
    optimizer.step()
    return loss.item()


def train(
    model: nn.Module,
    data: Tensor,
    steps: int,
    batch_size: int = 32,
    max_lr: float = 3e-3,
    warmup: int = 20,
    seed: int = 0,
) -> list[float]:
    """Trains in place; returns the loss at every step."""
    generator = torch.Generator().manual_seed(seed)
    # AdamW decouples weight decay from the gradient (Adam + L2 is not the
    # same thing, 09.05). betas are the defaults; only the LR is scheduled.
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        x, y = get_batch(data, model.context, batch_size, generator)
        losses.append(
            train_step(model, optimizer, x, y, lr_at(step, max_lr, warmup, steps))
        )
    return losses


def test_schedule_warms_up_then_decays():
    max_lr, warmup, total = 1e-3, 10, 100
    assert lr_at(0, max_lr, warmup, total) == max_lr / warmup
    assert lr_at(warmup, max_lr, warmup, total) == max_lr
    assert lr_at(total, max_lr, warmup, total) == 0.0
    values = [lr_at(s, max_lr, warmup, total) for s in range(total + 1)]
    assert all(a <= b for a, b in pairwise(values[: warmup + 1]))
    assert all(a >= b for a, b in pairwise(values[warmup:]))
    assert max(values) <= max_lr


def test_batches_are_windows_with_shifted_targets():
    data = torch.arange(100)
    x, y = get_batch(
        data, context=8, batch_size=4, generator=torch.Generator().manual_seed(0)
    )
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert int(y.max()) < 100


def test_model_gives_logits_and_a_loss():
    torch.manual_seed(0)
    model = TinyLM(vocab_size=30, context=16, d_model=32, n_heads=4, n_blocks=2)
    x = torch.randint(0, 30, (2, 16))
    y = torch.randint(0, 30, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 30)
    # An untrained model should be close to uniform: loss near ln(vocab).
    assert abs(loss.item() - math.log(30)) < 1.0
    assert model.head.weight is model.token.weight


def test_one_step_clips_gradients_and_changes_parameters():
    torch.manual_seed(0)
    model = TinyLM(vocab_size=30, context=16, d_model=32, n_heads=4, n_blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randint(0, 30, (4, 16))
    y = torch.randint(0, 30, (4, 16))
    # A loss scaled up by 1e4 has enormous gradients. After clipping to a
    # total norm of 1.0, what the optimiser saw must have norm <= 1.0.
    model.head.weight.data *= 1e2
    train_step(model, optimizer, x, y, lr=1e-3, max_norm=1.0)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    assert total_norm <= 1.0 + 1e-4
    assert any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, model.parameters(), strict=True)
    )


def test_gradients_do_not_accumulate_across_steps():
    torch.manual_seed(0)
    model = TinyLM(vocab_size=30, context=16, d_model=32, n_heads=4, n_blocks=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # no update: isolate .grad
    x = torch.randint(0, 30, (4, 16))
    y = torch.randint(0, 30, (4, 16))
    train_step(model, optimizer, x, y, lr=0.0, max_norm=1e9)
    first = model.position.weight.grad.clone()
    train_step(model, optimizer, x, y, lr=0.0, max_norm=1e9)
    torch.testing.assert_close(model.position.weight.grad, first)


def test_training_reaches_a_low_loss():
    torch.manual_seed(0)
    tok = CharTokeniser(CORPUS)
    data = tok.encode(CORPUS)
    model = TinyLM(tok.vocab_size)
    losses = train(model, data, steps=300)
    start = sum(losses[:10]) / 10
    end = sum(losses[-10:]) / 10
    assert start > 3.0, f"initial loss {start:.2f} is suspiciously low"
    assert end < 1.6, f"final loss {end:.2f}: the model did not learn"
