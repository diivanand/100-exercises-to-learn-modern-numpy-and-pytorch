# =============================================================================
#  07.05 -- In-place operations
# =============================================================================
#
#  A method ending in an underscore -- `add_`, `mul_`, `zero_`, `clamp_`,
#  `copy_` -- modifies its tensor in place and returns it. The same
#  operation without the underscore allocates a new tensor. Both are
#  useful; mixing them up is the bug.
#
#   * `x -= x.mean()` on a function's argument changes the CALLER's tensor.
#     A function that only means to compute something must not do that.
#   * `total = total + x` REBINDS the local name; the caller's `total` is
#     untouched. A function that means to accumulate into the caller's
#     buffer needs `total.add_(x)` or `total += x`.
#   * An in-place op on a leaf tensor that requires grad raises "a leaf
#     Variable that requires grad is being used in an in-place operation".
#     That is autograd protecting the value it needs for the backward pass.
#     Parameter updates go inside `torch.no_grad()` (08.03; Godoy, ch. 1
#     "no_grad": "One does not simply update parameters without no_grad").
#
#  In-place ops save memory and are how optimisers update weights; the price
#  is that every alias of the tensor sees the change.
#
#  TASK
#    Make `standardise` leave its input alone, make `accumulate` change the
#    caller's buffer, and make `halve_parameter` work on a leaf that
#    requires grad.
#
#  RUN IT
#    ./npt test 07_05
#
# =============================================================================

import torch


def standardise(x: torch.Tensor) -> torch.Tensor:
    """(x - mean) / std as a new tensor; `x` is not modified."""
    # TODO: both lines modify the caller's tensor. Compute a new one.
    x -= x.mean()
    x /= x.std()
    return x


def accumulate(total: torch.Tensor, x: torch.Tensor) -> None:
    """Add `x` into `total`, which the caller keeps."""
    # TODO: this rebinds the local name `total`; the caller sees nothing.
    total = total + x


def halve_parameter(p: torch.Tensor) -> None:
    """Halve a parameter in place without autograd objecting."""
    # TODO: autograd refuses an in-place change to a leaf that requires grad.
    # Tell it this is not part of any computation it should track.
    p.mul_(0.5)


def test_standardise_does_not_touch_its_input():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    before = x.clone()
    z = standardise(x)
    torch.testing.assert_close(x, before)
    torch.testing.assert_close(z.mean(), torch.tensor(0.0))
    torch.testing.assert_close(z.std(), torch.tensor(1.0))


def test_accumulate_changes_the_callers_buffer():
    total = torch.zeros(3)
    accumulate(total, torch.tensor([1.0, 2.0, 3.0]))
    accumulate(total, torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(total, torch.tensor([2.0, 3.0, 4.0]))


def test_halve_parameter_on_a_leaf():
    p = torch.tensor([2.0, 4.0], requires_grad=True)
    halve_parameter(p)
    torch.testing.assert_close(p.detach(), torch.tensor([1.0, 2.0]))
    assert p.requires_grad
    assert p.is_leaf
