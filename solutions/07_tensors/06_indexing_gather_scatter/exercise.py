# Solution -- 07.06 gather, scatter and masks
import torch


def pick_class_logit(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """For each row of `logits` (B, C), the entry at that row's target."""
    return logits.gather(1, targets[:, None]).squeeze(1)


def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, C) float tensor with a 1.0 at each row's target."""
    out = torch.zeros(targets.shape[0], num_classes, dtype=torch.float32)
    return out.scatter_(1, targets[:, None], 1.0)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the positions where `mask` is True, per row."""
    masked = x.masked_fill(~mask, 0.0)
    return masked.sum(dim=1) / mask.sum(dim=1)


def test_pick_class_logit():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
    targets = torch.tensor([1, 0, 2])
    torch.testing.assert_close(
        pick_class_logit(logits, targets), torch.tensor([0.9, 0.7, 0.4])
    )


def test_pick_class_logit_rectangular():
    # More rows than classes: a gather along the wrong dim cannot even index.
    logits = torch.arange(10.0).reshape(5, 2)
    targets = torch.tensor([1, 0, 1, 0, 1])
    torch.testing.assert_close(
        pick_class_logit(logits, targets), torch.tensor([1.0, 2.0, 5.0, 6.0, 9.0])
    )


def test_one_hot_matches_the_library():
    targets = torch.tensor([2, 0, 1, 2])
    expected = torch.nn.functional.one_hot(targets, 3).float()
    torch.testing.assert_close(one_hot(targets, 3), expected)


def test_one_hot_has_no_python_loop():
    import inspect

    assert "for " not in inspect.getsource(one_hot)


def test_masked_mean_ignores_padding():
    x = torch.tensor([[1.0, 2.0, 100.0], [4.0, 100.0, 100.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    torch.testing.assert_close(masked_mean(x, mask), torch.tensor([1.5, 4.0]))
