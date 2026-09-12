# Solution -- 10.04 collate_fn
import torch
from torch.nn.utils.rnn import pad_sequence
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
    lengths = torch.tensor([len(s) for s in sequences])
    # pad_sequence right-pads to the longest sequence in THIS batch, so the
    # padding cost is per batch, not per dataset. 0 is the padding id.
    tokens = pad_sequence(sequences, batch_first=True, padding_value=0)
    # A mask says which positions are real. Downstream code needs it: a mean
    # over the padded positions, or attention to them, would be wrong.
    mask = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
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
