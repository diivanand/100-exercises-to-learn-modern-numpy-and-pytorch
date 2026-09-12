# =============================================================================
#  12.02 -- NaN detection
# =============================================================================
#
#  A loss that becomes nan is the most common way a training run dies, and
#  by the time you see it in the log the cause is several steps upstream.
#  Godoy (ch. 6, on exploding gradients) shows the pattern: the loss reads
#  16985, then nan, and the gradients before it are already 1e26. The
#  causes are few and worth knowing by heart:
#
#   * log of zero. `torch.log(torch.softmax(x))` is the classic: for logits
#     around 100 the softmax underflows to exactly 0 for every class but
#     one, log(0) is -inf, and the gradient of -inf is nan. The fix is
#     `F.log_softmax` (Godoy ch. 5, "LogSoftmax"), or in general the
#     log-sum-exp trick: log Σ exp(xᵢ) = m + log Σ exp(xᵢ - m) with m = max x.
#     Never take the log of a probability you computed with exp.
#   * 0/0 and inf - inf, from a normalisation by a sum that can be zero or a
#     variance that can be zero (add eps, as LayerNorm does).
#   * sqrt or log of a slightly negative number that should have been zero.
#   * A learning rate that is too high: gradients grow geometrically.
#
#  Finding WHERE is the harder part, because nan propagates through every
#  operation after it. Two tools:
#
#      torch.autograd.set_detect_anomaly(True)
#
#  as a context manager (or globally) makes the backward pass raise at the
#  FIRST operation whose gradient contains nan, with a traceback of the
#  forward call that created it. It is slow -- use it to find the bug, not in
#  production -- and it is the reason the tests here run the backward pass
#  under it. And a small `check_finite(t, "name")` at the boundaries you
#  care about (the loss, the logits, the gradient norm), which turns a
#  silent nan into an error that says what and where.
#
#  TASK
#    Make `log_probabilities` numerically stable, make `check_finite` report
#    the count and positions of the bad entries, and check that the
#    gradients are finite under anomaly detection.
#
#  RUN IT
#    ./npt test 12_02
#
# =============================================================================

import torch
import torch.nn.functional as F


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """log softmax over the last axis, stable for logits of any magnitude."""
    # TODO: for logits of magnitude ~1000 the softmax is exactly 0 for all
    # but one class, and log(0) is -inf. Use F.log_softmax (or logsumexp).
    return torch.log(torch.softmax(logits, dim=-1))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of `targets` under `logits`."""
    log_probs = log_probabilities(logits)
    picked = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -picked.mean()


def check_finite(t: torch.Tensor, name: str) -> torch.Tensor:
    """Return `t`, or raise a ValueError that says which entries are not finite."""
    # TODO: torch.isnan misses inf. Use torch.isfinite, count the offenders
    # and include the first few positions (bad.nonzero()) in the message.
    if torch.isnan(t).any():
        raise ValueError(f"{name} has nan")
    return t


def test_log_probabilities_survive_huge_logits():
    logits = torch.tensor([[1000.0, 0.0, -1000.0]])
    lp = log_probabilities(logits)
    assert torch.isfinite(lp).all()
    torch.testing.assert_close(lp[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(lp[0, 1], torch.tensor(-1000.0))
    torch.testing.assert_close(lp[0, 2], torch.tensor(-2000.0))


def test_cross_entropy_matches_the_library_and_stays_finite():
    torch.manual_seed(0)
    logits = torch.randn(16, 5) * 300.0
    targets = torch.randint(0, 5, (16,))
    ours = cross_entropy(logits, targets)
    assert torch.isfinite(ours)
    torch.testing.assert_close(ours, F.cross_entropy(logits, targets))


def test_gradients_are_finite_under_anomaly_detection():
    torch.manual_seed(0)
    w = torch.randn(5, 3, requires_grad=True)
    x = torch.randn(4, 5) * 100.0
    targets = torch.tensor([0, 1, 2, 0])
    with torch.autograd.set_detect_anomaly(True):
        loss = cross_entropy(x @ w, targets)
        loss.backward()
    assert torch.isfinite(w.grad).all()


def test_check_finite_reports_where():
    t = torch.tensor([1.0, float("nan"), 2.0, float("inf")])
    try:
        check_finite(t, "loss")
    except ValueError as e:
        assert "loss" in str(e) and "2 non-finite" in str(e) and "[1]" in str(e)
    else:
        raise AssertionError("non-finite values were not reported")
    assert check_finite(torch.ones(3), "ok") is not None
