# Solution -- 12.02 NaN detection

import torch
import torch.nn.functional as F


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """log softmax over the last axis, stable for logits of any magnitude."""
    # log_softmax computes x - logsumexp(x), and logsumexp subtracts the row
    # maximum before exponentiating, so nothing overflows or underflows to
    # exactly zero. log(softmax(x)) computes the zero first and the log of it
    # second.
    return F.log_softmax(logits, dim=-1)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of `targets` under `logits`."""
    log_probs = log_probabilities(logits)
    picked = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -picked.mean()


def check_finite(t: torch.Tensor, name: str) -> torch.Tensor:
    """Return `t`, or raise a ValueError that says which entries are not finite."""
    bad = ~torch.isfinite(t)
    if bad.any():
        where = bad.nonzero()[:5].tolist()
        raise ValueError(
            f"{name} has {int(bad.sum())} non-finite entries, first at {where}"
        )
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
