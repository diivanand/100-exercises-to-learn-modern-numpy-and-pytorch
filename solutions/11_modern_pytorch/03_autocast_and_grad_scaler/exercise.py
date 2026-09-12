# Solution -- 11.03 Autocast and GradScaler

import torch
from torch import nn


def mixed_precision_forward(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, loss) with the forward pass in reduced precision.

    Autocast decides per operator: matmuls run in bfloat16 (or float16 on
    CUDA), reductions and the loss stay in float32. The parameters are never
    touched, which is the whole point -- a permanently half-precision model
    cannot accumulate small updates.
    """
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
    return logits, loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    device_type: str,
) -> float:
    """One optimisation step under autocast with loss scaling."""
    optimizer.zero_grad()
    _, loss = mixed_precision_forward(model, x, y, device_type)
    # Backward is OUTSIDE autocast (the docs are explicit about that). The
    # scaler multiplies the loss so that float16 gradients do not underflow,
    # then `step` unscales before the optimiser sees them and skips the step
    # if any gradient is inf/nan; `update` adapts the scale for next time.
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def make_data():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randint(0, 3, (64,))
    return x, y


def test_forward_runs_in_bfloat16_but_the_loss_is_float32():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
    x, y = make_data()
    logits, loss = mixed_precision_forward(model, x, y, "cpu")
    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_parameters_stay_in_float32():
    torch.manual_seed(0)
    model = nn.Linear(16, 3)
    x, y = make_data()
    mixed_precision_forward(model, x, y, "cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_train_step_applies_the_unscaled_gradient():
    torch.manual_seed(0)
    model = nn.Linear(16, 3)
    x, y = make_data()
    reference = nn.Linear(16, 3)
    reference.load_state_dict(model.state_dict())

    # What a plain float32 SGD step would do to the same weights.
    loss = nn.functional.cross_entropy(reference(x), y)
    loss.backward()
    with torch.no_grad():
        expected_weight = reference.weight - 0.1 * reference.weight.grad

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    train_step(model, optimizer, scaler, x, y, "cpu")

    # bfloat16 matmuls in the forward pass perturb the gradient a little;
    # a scale factor of 65536 left in by mistake perturbs it a lot.
    torch.testing.assert_close(model.weight, expected_weight, rtol=0.05, atol=0.02)
    assert scaler.get_scale() == 65536.0
