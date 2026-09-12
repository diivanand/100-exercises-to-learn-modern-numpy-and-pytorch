# =============================================================================
#  08.02 -- Gradients accumulate
# =============================================================================
#
#  `.grad` is not SET by `backward()`, it is ADDED TO. Call backward twice
#  and the second gradient lands on top of the first. This is a feature --
#  it is how gradient accumulation over micro-batches works (09.10) -- and
#  it is the single most common bug in a hand-written training loop:
#  forgetting to zero the gradients, so every step uses the sum of all the
#  gradients so far and the parameters shoot off.
#
#  Godoy, ch. 1 "zero_": "we need to zero the gradients afterward. And
#  that's what zero_() is good for." Since PyTorch 2.0 the preferred reset
#  is `p.grad = None`, which is what `optimizer.zero_grad()` does by default
#  (`set_to_none=True`): the next backward allocates a fresh tensor rather
#  than writing zeros into an old one, and a None grad cannot be mistaken
#  for a real one.
#
#  TASK
#    Make `sgd` produce the same parameter as a loop that recomputes the
#    gradient from scratch each step.
#
#  RUN IT
#    ./npt test 08_02
#
# =============================================================================

import torch


def sgd(x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> float:
    """Fit y = w x by plain gradient descent, returning w."""
    w = torch.zeros((), requires_grad=True)
    for _ in range(steps):
        loss = ((w * x - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        # TODO: nothing resets w.grad, so each step applies the SUM of every
        # gradient so far. Reset it after the update.
    return w.item()


def _reference(x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> float:
    # The closed-form gradient, recomputed from scratch each step.
    w = 0.0
    for _ in range(steps):
        grad = (2 * (w * x - y) * x).mean().item()
        w -= lr * grad
    return w


def test_sgd_converges():
    torch.manual_seed(0)
    x = torch.randn(64)
    y = 3.0 * x
    assert abs(sgd(x, y, lr=0.1, steps=100) - 3.0) < 1e-3


def test_sgd_matches_a_fresh_gradient_each_step():
    torch.manual_seed(1)
    x = torch.randn(64)
    y = -1.5 * x
    for steps in (1, 2, 5):
        got = sgd(x, y, lr=0.05, steps=steps)
        want = _reference(x, y, lr=0.05, steps=steps)
        assert abs(got - want) < 1e-5, f"after {steps} steps: {got} vs {want}"
