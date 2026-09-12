# =============================================================================
#  13.01 -- Capstone: tokeniser and embeddings
# =============================================================================
#
#  The capstone builds a small language model from parts you have already
#  met: tensors and indexing (chapter 07), autograd (08), nn.Module and the
#  training loop (09), batching (10), scaled-dot-product attention and
#  device-agnostic code (11). Five exercises, one model. Each file is
#  self-contained, so the pieces you finish here reappear in the next ones,
#  marked "unchanged", and you only work on the new part.
#
#  A language model predicts the next token of a sequence. Before there can
#  be a next token there have to be tokens, and before the model can compute
#  with them each token has to become a vector. That is this exercise.
#
#  TOKENS. Real systems use sub-word tokenisers (byte-pair encoding, which
#  Kneusel PDL ch. 18 walks through); we use CHARACTERS, which needs no
#  training and gives a vocabulary of about sixty symbols. A tokeniser is two
#  functions, encode and decode, and the only property that matters is that
#  decode(encode(text)) == text. Volpe et al. ch. 8, "Tokenization and
#  Conversion to Embeddings", does the same with words.
#
#  EMBEDDINGS. `nn.Embedding(vocab, d)` is a lookup table: a (vocab, d) matrix
#  and integer indices in, rows out. It is a linear layer applied to a one-hot
#  vector, without materialising the one-hot. Gradients flow into exactly the
#  rows that were looked up.
#
#  POSITIONS. Attention (13.02) is a set operation: it has no idea which token
#  came first. So the position is added in explicitly, as a second embedding
#  table with one learned row per position (GPT-2 does this; Godoy ch. 9,
#  "Positional Encoding", uses fixed sinusoids instead -- either works at this
#  scale). Token and position vectors are summed, not concatenated: the model
#  learns to keep them apart in different directions of the same space.
#
#  TARGETS. The training signal is the text itself, shifted by one: for the
#  window `data[s : s + T]` the target is `data[s + 1 : s + T + 1]`, so the
#  target at every position is the input at the next one (Cautaerts ch. 14,
#  "Preprocessing").
#
#  TASK
#    Make the tokeniser map characters to a contiguous 0..V-1 range in sorted
#    order and back, make the embedding depend on position and refuse
#    sequences longer than its context, and fix the target window.
#
#  RUN IT
#    ./npt test 13_01
#
# =============================================================================

import torch
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


class CharTokeniser:
    """Maps each distinct character of a text to a small integer, and back."""

    def __init__(self, text: str) -> None:
        # TODO: ids must be assigned in sorted character order, so that two
        # tokenisers built from texts with the same alphabet agree and the
        # ids are a contiguous 0..V-1 range. First-seen order is neither.
        self.chars: list[str] = list(dict.fromkeys(text))
        self._to_id: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> Tensor:
        # A KeyError from the dict is the right failure for a character the
        # tokeniser has never seen: loud, early, and naming the character.
        return torch.tensor([self._to_id[ch] for ch in text], dtype=torch.long)

    def decode(self, ids: Tensor) -> str:
        # TODO: characters are joined with nothing in between, not spaces.
        return " ".join(self.chars[int(i)] for i in ids)


class TokenAndPositionEmbedding(nn.Module):
    """Token identity plus position, summed into one vector per position."""

    def __init__(self, vocab_size: int, context: int, d_model: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        # One learned vector per position, up to the context length. This is
        # the simplest positional scheme (GPT-2 uses it); the sinusoidal one
        # in Godoy ch. 9 "Positional Encoding" needs no parameters but is no
        # better at this scale.
        self.position = nn.Embedding(context, d_model)
        self.context = context

    def forward(self, idx: Tensor) -> Tensor:
        length = idx.shape[1]
        # TODO: raise ValueError if `length` exceeds the context: the position
        # table has no row for position `context`, and the IndexError it would
        # give deep inside nn.Embedding is a far worse message.
        #
        # TODO: add the position embedding. Look up `torch.arange(length)` in
        # `self.position` and add it; (T, D) broadcasts against (B, T, D).
        return self.token(idx)


def make_example(data: Tensor, context: int, start: int) -> tuple[Tensor, Tensor]:
    """One training pair: `context` tokens, and the same window shifted by one."""
    x = data[start : start + context]
    # TODO: the target is the input shifted by ONE position: y[t] is the token
    # that follows x[t]. As written the model is asked to predict its input.
    y = data[start : start + context]
    return x, y


def test_tokeniser_round_trips():
    tok = CharTokeniser(CORPUS)
    ids = tok.encode(CORPUS)
    assert ids.dtype == torch.long
    assert ids.shape == (len(CORPUS),)
    assert tok.decode(ids) == CORPUS


def test_ids_are_a_contiguous_range_in_alphabet_order():
    tok = CharTokeniser("banana band")
    assert tok.chars == [" ", "a", "b", "d", "n"]
    assert tok.vocab_size == 5
    assert tok.encode("band").tolist() == [2, 1, 4, 3]
    assert tok.decode(torch.tensor([2, 1, 4, 3])) == "band"


def test_unknown_characters_are_an_error():
    import pytest

    tok = CharTokeniser("abc")
    with pytest.raises(KeyError):
        tok.encode("abz")


def test_embedding_shape_and_position_dependence():
    torch.manual_seed(0)
    emb = TokenAndPositionEmbedding(vocab_size=10, context=8, d_model=16)
    idx = torch.tensor([[3, 3, 3, 3]])
    out = emb(idx)
    assert out.shape == (1, 4, 16)
    # Same token at different positions must give different vectors,
    # otherwise the model has no way of knowing which comes first.
    assert not torch.allclose(out[0, 0], out[0, 1])
    # And the same token at the same position must give the same vector
    # regardless of what else is in the batch.
    other = emb(torch.tensor([[3, 7, 1, 9]]))
    torch.testing.assert_close(out[0, 0], other[0, 0])


def test_embedding_refuses_sequences_longer_than_its_context():
    import pytest

    emb = TokenAndPositionEmbedding(vocab_size=10, context=4, d_model=8)
    with pytest.raises(ValueError):
        emb(torch.zeros(1, 5, dtype=torch.long))


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(20)
    x, y = make_example(data, context=6, start=3)
    assert x.tolist() == [3, 4, 5, 6, 7, 8]
    assert y.tolist() == [4, 5, 6, 7, 8, 9]
    # The target at every position is the input at the next one: that is the
    # entire training signal of a language model.
    assert torch.equal(x[1:], y[:-1])
