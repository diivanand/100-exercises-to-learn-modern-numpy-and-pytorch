# =============================================================================
#  09.04 -- Losses take logits
# =============================================================================
#
#  A classifier's last layer produces LOGITS: unbounded real numbers, one per
#  class. Probabilities are softmax (or sigmoid) of the logits. The question
#  is where that softmax lives, and the answer in PyTorch is: inside the loss.
#
#      nn.CrossEntropyLoss / F.cross_entropy           takes logits (N, C)
#                                                      and class INDICES (N,)
#      nn.BCEWithLogitsLoss / F.binary_cross_entropy_with_logits
#                                                      takes logits and 0/1 targets
#
#  Godoy is emphatic about it (ch. 3, "BCEWithLogitsLoss": "you should NOT add
#  a sigmoid as the last layer of your model when using this loss function"),
#  and ch. 5, "Loss", walks through logits, softmax, log-softmax and negative
#  log-likelihood to show that cross_entropy IS log_softmax followed by
#  nll_loss, in one step.
#
#  Why one step? Two reasons.
#
#   1. Numerical: log(softmax(x)) computed as two operations overflows for
#      logits above about 88 (exp(x) in float32) and gives log(0) = -inf for
#      confident wrong answers. The fused version uses the log-sum-exp trick
#      and is exact.
#
#   2. Double application: if the model already ends in a softmax and you
#      pass its output to cross_entropy, the loss softmaxes AGAIN. The values
#      are then squashed into [0, 1] before the second softmax, so the model
#      can never be confident, the loss never goes below about log(C) - 1,
#      and nothing errors. It just trains badly, for a reason nobody can see.
#
#  The older pairs -- nn.Sigmoid + nn.BCELoss, nn.LogSoftmax + nn.NLLLoss --
#  still exist. Prefer the fused ones unless you need the probabilities
#  for something else.
#
#  Imbalanced classes: `pos_weight` in BCEWithLogitsLoss multiplies the
#  positive-class terms (Godoy ch. 3, "Imbalanced Dataset"); `label_smoothing`
#  in cross_entropy mixes a little uniform target into the one-hot one.
#
#  TASK
#    Make all three functions compute their loss from the raw logits, with
#    the fused functions.
#
#  RUN IT
#    ./npt test 09_04
#
# =============================================================================

import torch
import torch.nn.functional as F


def multiclass_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy over class indices. `logits` is (N, C), raw model output."""
    # TODO: cross_entropy applies softmax itself. This applies it twice.
    probabilities = logits.softmax(dim=-1)
    return F.cross_entropy(probabilities, targets, label_smoothing=label_smoothing)


def binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy from raw logits."""
    # TODO: sigmoid(-10000) is exactly 0.0 in float32, and log(0) is not a
    # number. The fused function never forms the probability.
    return F.binary_cross_entropy(torch.sigmoid(logits), targets)


def imbalanced_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, negatives_per_positive: float
) -> torch.Tensor:
    """Binary cross-entropy that up-weights the (rarer) positive class."""
    # TODO: `weight` scales EVERY term by a per-element factor; `pos_weight`
    # scales only the positive-class terms, by a per-class factor.
    weight = torch.full_like(logits, negatives_per_positive)
    return F.binary_cross_entropy(torch.sigmoid(logits), targets, weight=weight)


def test_multiclass_matches_the_definition():
    torch.manual_seed(0)
    logits = torch.randn(6, 4)
    targets = torch.tensor([0, 3, 1, 2, 3, 0])
    expected = -F.log_softmax(logits, dim=-1)[torch.arange(6), targets].mean()
    torch.testing.assert_close(multiclass_loss(logits, targets), expected)


def test_multiclass_is_not_softmaxed_twice():
    # Very confident, correct logits: the loss must be close to zero.
    logits = torch.tensor([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
    targets = torch.tensor([0, 1])
    assert multiclass_loss(logits, targets).item() < 1e-6


def test_label_smoothing_changes_the_value():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    plain = multiclass_loss(logits, targets)
    smoothed = multiclass_loss(logits, targets, label_smoothing=0.1)
    assert not torch.isclose(plain, smoothed)
    expected = 0.9 * plain + 0.1 * (-F.log_softmax(logits, dim=-1).mean())
    torch.testing.assert_close(smoothed, expected)


def test_binary_matches_the_definition():
    torch.manual_seed(0)
    logits = torch.randn(10)
    targets = (torch.rand(10) > 0.5).float()
    p = torch.sigmoid(logits)
    expected = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).mean()
    torch.testing.assert_close(binary_loss(logits, targets), expected)


def test_binary_does_not_saturate():
    # A wrong answer given with a logit of 10000 should cost about 10000 nats.
    # sigmoid(-10000) is exactly 0.0 in float32, and log(0) is where a
    # sigmoid-then-BCELoss pipeline clamps or explodes.
    logits = torch.tensor([-10000.0])
    targets = torch.tensor([1.0])
    assert abs(binary_loss(logits, targets).item() - 10000.0) < 1.0


def test_pos_weight_scales_positive_terms_only():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    plain = binary_loss(logits, targets)
    weighted = imbalanced_binary_loss(logits, targets, negatives_per_positive=3.0)
    # Both terms are log(2) unweighted; the positive one is tripled.
    torch.testing.assert_close(plain, torch.tensor(0.6931), atol=1e-3, rtol=0)
    torch.testing.assert_close(weighted, torch.tensor(2 * 0.6931), atol=1e-3, rtol=0)
