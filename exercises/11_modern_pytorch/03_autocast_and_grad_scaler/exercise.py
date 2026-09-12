# =============================================================================
#  11.03 -- Autocast and GradScaler
# =============================================================================
#
#  A 4090 does float32 matrix multiplies at about 80 TFLOP/s and bfloat16 or
#  float16 ones at roughly twice that, with half the memory traffic. Mixed
#  precision means: run the expensive operators in the narrow type, keep the
#  parameters, the optimiser state and the loss in float32. You do not do the
#  casting by hand. You wrap the forward pass:
#
#      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
#          logits = model(x)
#          loss = loss_fn(logits, y)
#      loss.backward()        # outside the context
#
#  Autocast keeps a list of operators that are safe in the narrow type
#  (matmul, conv) and a list that must stay in float32 (softmax, losses,
#  norms) and casts inputs per operator. The parameters are untouched, so
#  `model.parameters()` are still float32 afterwards -- which is what a
#  permanently casted `model.to(torch.bfloat16)` breaks: an update of 1e-4 to
#  a weight of 1.0 vanishes in bfloat16's 8 bits of mantissa.
#
#  Two details from the documentation (https://docs.pytorch.org/docs/2.14/amp.html):
#
#   * The spelling. `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`
#     are deprecated since 2.4; the current API takes the device type as its
#     first argument: `torch.amp.autocast("cuda", ...)`, `torch.amp.GradScaler("cuda")`.
#     On the CPU the only narrow type is bfloat16.
#
#   * "autocast should wrap only the forward pass(es) of your network,
#     including the loss computation(s). Backward passes under autocast are
#     not recommended." The backward runs in whatever type the forward used.
#
#  GradScaler is for float16 specifically. Its 5-bit exponent underflows for
#  gradients below about 6e-8, so the scaler multiplies the loss by a large
#  factor (65536 to start), which scales every gradient by the same factor;
#  then it divides the gradients back down before the optimiser applies them,
#  skips the step if any became inf/nan, and adjusts the factor:
#
#      scaler.scale(loss).backward()
#      scaler.step(optimizer)      # unscales, checks for inf, steps
#      scaler.update()             # grows or shrinks the scale
#
#  bfloat16 has float32's exponent range and does not need scaling; on a
#  4090 use bfloat16 and skip the scaler unless you have a reason. The
#  pattern is still worth knowing, because it is what you will read in every
#  training script written before bfloat16 hardware was common, and because
#  it has a trap: calling `optimizer.step()` instead of `scaler.step(optimizer)`
#  applies the SCALED gradient, 65536 times too large, and the loss becomes nan
#  on the next step with no error anywhere.
#
#  The tests run on the CPU with bfloat16, so they exercise the same code path
#  you will use on the 4090 with "cuda" in place of "cpu".
#
#  TASK
#    Make `mixed_precision_forward` use autocast rather than casting the model,
#    and make `train_step` go through the scaler for the optimiser step.
#
#  RUN IT
#    ./npt test 11_03
#
# =============================================================================

import torch
from torch import nn


def mixed_precision_forward(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device_type: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, loss) with the forward pass in reduced precision."""
    # TODO: this casts the parameters themselves to bfloat16, permanently,
    # and computes the loss in bfloat16 too. Use torch.amp.autocast around
    # the forward pass and the loss instead; leave the model alone.
    model.to(torch.bfloat16)
    logits = model(x.to(torch.bfloat16))
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
    scaler.scale(loss).backward()
    # TODO: the loss was scaled, so the gradients are scaled. Stepping the
    # optimiser directly applies them 65536 times too large. The scaler has
    # to do the step (it unscales first) and then be updated.
    optimizer.step()
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
