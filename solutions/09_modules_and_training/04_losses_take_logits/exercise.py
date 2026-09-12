# Solution -- 09.04 Losses take logits
import torch
import torch.nn.functional as F


def multiclass_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy over class indices. `logits` is (N, C), raw model output."""
    # cross_entropy = log_softmax + nll_loss, in one numerically stable step.
    # Feeding it probabilities would apply softmax a second time.
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy from raw logits."""
    # The sigmoid lives inside the loss, where it is computed as a log-sum-exp
    # that cannot saturate to log(0).
    return F.binary_cross_entropy_with_logits(logits, targets)


def imbalanced_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, negatives_per_positive: float
) -> torch.Tensor:
    """Binary cross-entropy that up-weights the (rarer) positive class."""
    pos_weight = torch.tensor([negatives_per_positive], dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


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
