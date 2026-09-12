# Solution -- 09.10 Gradient clipping and accumulation
import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]


def accumulate_gradients(model: nn.Module, micro_batches: list[Batch]) -> None:
    """Leave in .grad the gradient of the MEAN loss over all micro-batches."""
    k = len(micro_batches)
    for x, y in micro_batches:
        loss = nn.functional.mse_loss(model(x), y)
        # backward() ADDS to .grad. Summing k mean-losses gives k times the
        # gradient of the mean over the whole batch, so each is scaled by 1/k.
        (loss / k).backward()


def clip_and_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
) -> float:
    """Clip the global gradient norm, step, and return the norm BEFORE clipping."""
    # clip_grad_norm_ treats all gradients as one vector and scales the whole
    # vector down if its L2 norm exceeds max_norm. Clamping each element
    # separately would change the gradient's direction.
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad()
    return total_norm.item()


def make_data(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    y = x @ torch.tensor([[2.0], [-1.0], [0.5]]) + 10.0
    return x, y


def grads_of(model: nn.Module) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters()]


def test_accumulated_gradient_equals_full_batch_gradient():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    x, y = make_data()

    nn.functional.mse_loss(model(x), y).backward()
    full = grads_of(model)
    model.zero_grad()

    micro = [(x[i : i + 16], y[i : i + 16]) for i in range(0, 64, 16)]
    accumulate_gradients(model, micro)
    torch.testing.assert_close(grads_of(model), full)


def test_clipping_bounds_the_global_norm_and_reports_the_original():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    x, y = make_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # step must not move anything
    nn.functional.mse_loss(model(x), y).backward()
    grads = grads_of(model)
    original_norm = torch.sqrt(sum((g**2).sum() for g in grads)).item()
    assert original_norm > 1.0

    reported = clip_and_step(model, optimizer, max_norm=1.0)
    assert abs(reported - original_norm) < 1e-4
    assert all(p.grad is None for p in model.parameters())


def test_clipping_preserves_direction():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    x, y = make_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = [p.detach().clone() for p in model.parameters()]
    nn.functional.mse_loss(model(x), y).backward()
    grads = grads_of(model)
    clip_and_step(model, optimizer, max_norm=1.0)
    after = [p.detach() for p in model.parameters()]
    steps = [b - a for b, a in zip(before, after, strict=True)]
    # The step is the gradient scaled by one number: parallel, with norm 1.
    step_norm = torch.sqrt(sum((s**2).sum() for s in steps)).item()
    assert abs(step_norm - 1.0) < 1e-4
    scale = steps[0].flatten()[0] / grads[0].flatten()[0]
    for s, g in zip(steps, grads, strict=True):
        torch.testing.assert_close(s, g * scale)
