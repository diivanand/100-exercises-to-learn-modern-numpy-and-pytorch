# =============================================================================
#  07.06 -- gather, scatter and masks
# =============================================================================
#
#  Plain indexing covers most of what you need; the rest is three verbs.
#
#   * `x.gather(dim, index)` picks one element per position along `dim`,
#     with `index` the same shape as the result. Picking the logit of the
#     correct class for every row of a (B, C) tensor is
#     `logits.gather(1, targets[:, None]).squeeze(1)`, which is exactly what
#     a cross-entropy loss does before the log-sum-exp (09.04).
#   * `x.scatter_(dim, index, src)` is the inverse: writes into the given
#     positions. `scatter_add_` accumulates, and is how you build a one-hot
#     encoding or a histogram on the GPU without a loop.
#   * `torch.where(mask, a, b)` and `x.masked_fill(mask, value)` choose
#     without branching, which matters because `if` on a tensor cannot work
#     for more than one element at a time (02.05 made the same point for
#     NumPy).
#
#  The dim argument is the one that is INDEXED; get it wrong and gather
#  either raises "index out of bounds" or, worse, returns a valid tensor of
#  the wrong numbers.
#
#  TASK
#    Fix the dimension in `pick_class_logit`, build `one_hot` with a
#    scatter, and make `masked_mean` ignore the padded positions.
#
#  RUN IT
#    ./npt test 07_06
#
# =============================================================================

import torch


def pick_class_logit(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """For each row of `logits` (B, C), the entry at that row's target."""
    # TODO: the dimension being indexed is the class axis, dim 1.
    return logits.gather(0, targets[:, None]).squeeze(1)


def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, C) float tensor with a 1.0 at each row's target."""
    # TODO: replace the loop with a single scatter_ along dim 1.
    out = torch.zeros(targets.shape[0], num_classes, dtype=torch.float32)
    for row, t in enumerate(targets.tolist()):
        out[row, t] = 1.0
    return out


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the positions where `mask` is True, per row."""
    # TODO: this averages over every position, padding included. Zero the
    # masked-out entries and divide by the number of real ones.
    return x.mean(dim=1)


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
