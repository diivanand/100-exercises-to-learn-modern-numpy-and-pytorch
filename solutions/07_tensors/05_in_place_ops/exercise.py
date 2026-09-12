# Solution -- 07.05 In-place operations
import torch


def standardise(x: torch.Tensor) -> torch.Tensor:
    """(x - mean) / std as a new tensor; `x` is not modified."""
    return (x - x.mean()) / x.std()


def accumulate(total: torch.Tensor, x: torch.Tensor) -> None:
    """Add `x` into `total`, which the caller keeps."""
    total.add_(x)


def halve_parameter(p: torch.Tensor) -> None:
    """Halve a parameter in place without autograd objecting."""
    with torch.no_grad():
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
