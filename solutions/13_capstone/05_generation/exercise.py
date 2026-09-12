# Solution -- 13.05 Capstone: generation

import math

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


# --- from 13.01 to 13.04, unchanged -------------------------------------------
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


class TinyLM(nn.Module):
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
        self.head.weight = self.token.weight
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
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


def get_batch(
    data: Tensor, context: int, batch_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, len(data) - context, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + 1 : s + context + 1] for s in starts])
    return x, y


def lr_at(
    step: int, max_lr: float, warmup: int, total: int, min_lr: float = 0.0
) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(
    model: nn.Module, data: Tensor, steps: int, batch_size: int = 32, seed: int = 0
) -> None:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    model.train()
    for step in range(steps):
        x, y = get_batch(data, model.context, batch_size, generator)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, 3e-3, 20, steps)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# --- this exercise ------------------------------------------------------------
def next_token_logits(
    model: TinyLM, idx: Tensor, temperature: float, top_k: int | None
) -> Tensor:
    """Logits for the next token of each sequence in `idx`, shaped and filtered."""
    # The model has positional embeddings for `context` positions only, so
    # feed it the last `context` tokens. Everything earlier is forgotten:
    # that is the price of a fixed context window.
    logits, _ = model(idx[:, -model.context :])
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        # Keep the k largest logits per row; set the rest to -inf so that
        # softmax gives them exactly zero probability. The threshold is a
        # column vector (B, 1) so it compares row by row.
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return logits


@torch.inference_mode()
def generate(
    model: TinyLM,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Extend each row of `idx` by sampling `max_new_tokens` tokens."""
    model.eval()
    for _ in range(max_new_tokens):
        if temperature == 0.0:
            logits, _ = model(idx[:, -model.context :])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(
                next_token_logits(model, idx, temperature, top_k), dim=-1
            )
            # torch.multinomial draws from the rows of `probs`. The generator
            # is what makes the draw reproducible; without it, the global RNG
            # is used and two calls with the same inputs differ.
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def _untrained_model(vocab_size: int = 20, context: int = 8) -> TinyLM:
    torch.manual_seed(0)
    return TinyLM(vocab_size, context=context, d_model=16, n_heads=2, n_blocks=1)


def test_greedy_decoding_is_the_argmax_at_every_step():
    model = _untrained_model()
    prompt = torch.tensor([[1, 2, 3]])
    out = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :3], prompt)
    # Replay: each generated token must be the argmax of the model's logits
    # given everything before it. `out` came from inference_mode, so it is an
    # inference tensor and may only be used under inference_mode (08.03).
    with torch.inference_mode():
        for t in range(3, 9):
            logits, _ = model(out[:, max(0, t - model.context) : t])
            assert int(out[0, t]) == int(logits[0, -1].argmax())


def test_generation_runs_past_the_context_length():
    model = _untrained_model(context=8)
    out = generate(model, torch.tensor([[0, 1]]), max_new_tokens=20, temperature=0.0)
    assert out.shape == (1, 22)


def test_sampling_is_reproducible_with_a_generator():
    model = _untrained_model()
    prompt = torch.tensor([[4, 5]])
    a = generate(model, prompt, 30, generator=torch.Generator().manual_seed(7))
    b = generate(model, prompt, 30, generator=torch.Generator().manual_seed(7))
    c = generate(model, prompt, 30, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_very_low_temperature_is_greedy():
    model = _untrained_model()
    prompt = torch.tensor([[4, 5, 6]])
    greedy = generate(model, prompt, 10, temperature=0.0)
    cold = generate(
        model, prompt, 10, temperature=1e-4, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(greedy, cold)


def test_top_k_never_samples_outside_the_top_k():
    model = _untrained_model()
    prompt = torch.tensor([[1, 2], [3, 4]])  # two rows: thresholds differ per row
    k = 3
    out = generate(model, prompt, 25, top_k=k, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        for t in range(2, out.shape[1]):
            logits, _ = model(out[:, max(0, t - model.context) : t])
            top = torch.topk(logits[:, -1], k).indices
            for row in range(2):
                assert int(out[row, t]) in top[row].tolist()
    # And it must actually SAMPLE among the k, not always take the best.
    greedy = generate(model, prompt, 25, temperature=0.0)
    assert not torch.equal(out, greedy)


def test_a_trained_model_continues_the_corpus():
    torch.manual_seed(0)
    tok = CharTokeniser(CORPUS)
    data = tok.encode(CORPUS)
    model = TinyLM(tok.vocab_size)
    train(model, data, steps=500)  # about 4 s on a laptop CPU
    prompt = "Alice was beginning to get very tired of sitting by her "
    out = generate(model, tok.encode(prompt)[None, :], 40, temperature=0.0)
    text = tok.decode(out[0])
    assert text.startswith(prompt)
    assert set(text) <= set(tok.chars)
    # 500 steps over a 1.6 KB corpus is enough to memorise it: the greedy
    # continuation of the opening line is the opening line.
    assert text[len(prompt) :].startswith("sister on the bank")
