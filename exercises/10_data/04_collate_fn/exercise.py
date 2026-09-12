# =============================================================================
#  10.04 -- collate_fn
# =============================================================================
#
#  Between fetching the items of a batch and handing you a batch, the
#  DataLoader calls a COLLATE function on the list of items. The default one
#  stacks tensors along a new first dimension, recursing into tuples and
#  dicts. It works whenever every item has the same shape -- which is to say
#  it works for images and tabular rows, and fails with
#
#      RuntimeError: stack expects each tensor to be equal size
#
#  the moment your items are sentences, audio clips, point clouds, or
#  anything else of varying length.
#
#  `collate_fn=` replaces it. Yours receives the list of items exactly as
#  `__getitem__` returned them and returns whatever your training step wants.
#  For sequences the standard answer is to PAD to the longest item in the
#  batch (`torch.nn.utils.rnn.pad_sequence(..., batch_first=True)`) and to
#  return, alongside the padded tensor, the lengths and a boolean MASK
#  saying which positions are real. Padding to the longest in the batch,
#  rather than in the dataset, keeps the waste per batch; the mask is what
#  lets a loss or an attention layer ignore the padding (13.02).
#
#  TASK
#    Write `pad_collate` so that ragged sequences batch into a padded tensor
#    with lengths and a mask.
#
#  RUN IT
#    ./npt test 10_04
#
# =============================================================================

import torch
from torch.nn.utils.rnn import pad_sequence  # noqa: F401  (you will need it)
from torch.utils.data import DataLoader, Dataset


class Sentences(Dataset):
    """Token-id sequences of varying length, with a label each."""

    def __init__(self, n: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            length = int(torch.randint(1, 9, (1,), generator=generator))
            tokens = torch.randint(1, 50, (length,), generator=generator)
            label = torch.tensor(length % 2)
            self.items.append((tokens, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack ragged sequences into (tokens, lengths, mask, labels)."""
    sequences, labels = zip(*batch, strict=True)
    # TODO: torch.stack needs equal shapes. Pad with pad_sequence, and
    # compute the lengths and the mask from the unpadded sequences.
    tokens = torch.stack(sequences)
    lengths = torch.full((len(batch),), tokens.shape[1])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, lengths, mask, torch.stack(labels)


def test_collate_pads_to_the_longest_in_the_batch():
    batch = [
        (torch.tensor([5, 6, 7]), torch.tensor(1)),
        (torch.tensor([8]), torch.tensor(0)),
    ]
    tokens, lengths, mask, labels = pad_collate(batch)
    torch.testing.assert_close(tokens, torch.tensor([[5, 6, 7], [8, 0, 0]]))
    torch.testing.assert_close(lengths, torch.tensor([3, 1]))
    torch.testing.assert_close(
        mask, torch.tensor([[True, True, True], [True, False, False]])
    )
    torch.testing.assert_close(labels, torch.tensor([1, 0]))


def test_loader_uses_the_collate_fn():
    loader = DataLoader(Sentences(20), batch_size=4, collate_fn=pad_collate)
    for tokens, lengths, mask, labels in loader:
        assert tokens.ndim == 2 and tokens.shape[0] == 4
        assert tokens.shape[1] == int(lengths.max())
        assert mask.shape == tokens.shape
        assert mask.sum(dim=1).tolist() == lengths.tolist()
        assert labels.shape == (4,)


def test_padding_never_hides_real_tokens():
    ds = Sentences(20)
    loader = DataLoader(ds, batch_size=5, collate_fn=pad_collate)
    seen = []
    for tokens, lengths, mask, _ in loader:
        for row, length in zip(tokens, lengths, strict=True):
            seen.append(row[:length])
        assert torch.all(tokens[~mask] == 0)
    for (original, _), got in zip(ds.items, seen, strict=True):
        torch.testing.assert_close(got, original)
