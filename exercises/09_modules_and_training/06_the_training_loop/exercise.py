# =============================================================================
#  09.06 -- The training loop
# =============================================================================
#
#  Godoy's ch. 2 ("Rethinking the Training Loop") makes one move that is
#  worth copying everywhere: the body of the loop becomes a FUNCTION built by
#  a higher-order function that closes over the model, the loss and the
#  optimiser ("Training Step"):
#
#      def make_train_step(model, loss_fn, optimizer):
#          def train_step(x, y):
#              model.train()
#              loss = loss_fn(model(x), y)
#              loss.backward()
#              optimizer.step()
#              optimizer.zero_grad()
#              return loss.item()
#          return train_step
#
#  The loop itself is then two nested `for`s -- epochs, mini-batches -- with
#  nothing in them that can go wrong ("Mini-Batch Inner Loop").
#
#  The EVALUATION step is the same shape with two switches flipped, and they
#  are independent switches (Godoy ch. 2, "Evaluation"):
#
#   * `model.eval()` changes what some layers DO: Dropout stops dropping,
#     BatchNorm uses its running statistics (09.07). A model evaluated in
#     train mode gives a different answer every time you ask.
#
#   * `torch.inference_mode()` (or `torch.no_grad()`) changes what autograd
#     RECORDS: nothing. Without it every activation of every layer is kept
#     alive for a backward pass that never comes, which roughly doubles the
#     memory of evaluation for no reason -- and on a GPU that is the
#     difference between a batch that fits and one that does not.
#
#  `.item()` on the loss is deliberate too: it pulls one number out of the
#  graph so the Python list of losses does not keep every graph alive.
#
#  TASK
#    Make `make_eval_step` evaluate in eval mode with autograd off.
#
#  RUN IT
#    ./npt test 09_06
#
# =============================================================================

from collections.abc import Callable, Iterable

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]
Step = Callable[[torch.Tensor, torch.Tensor], float]


def make_train_step(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> Step:
    """Return a function that performs one optimisation step on a mini-batch."""

    def train_step(x: torch.Tensor, y: torch.Tensor) -> float:
        model.train()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


def make_eval_step(model: nn.Module, loss_fn: Callable[..., torch.Tensor]) -> Step:
    """Return a function that measures the loss on a mini-batch, nothing more."""

    def eval_step(x: torch.Tensor, y: torch.Tensor) -> float:
        # TODO: this evaluates with dropout still active and with autograd
        # recording every activation. Flip both switches.
        return loss_fn(model(x), y).item()

    return eval_step


def fit(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[Batch],
    val_batches: Iterable[Batch],
    epochs: int,
) -> dict[str, list[float]]:
    """The whole loop: epochs of mini-batches, then a validation pass."""
    train_step = make_train_step(model, loss_fn, optimizer)
    eval_step = make_eval_step(model, loss_fn)
    history: dict[str, list[float]] = {"train": [], "val": []}
    for _ in range(epochs):
        train_losses = [train_step(x, y) for x, y in train_batches]
        val_losses = [eval_step(x, y) for x, y in val_batches]
        history["train"].append(sum(train_losses) / len(train_losses))
        history["val"].append(sum(val_losses) / len(val_losses))
    return history


def make_batches(n: int, batch_size: int, seed: int) -> list[Batch]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=generator)
    y = (x[:, :1] * 3.0 - x[:, 1:] + 0.1 * torch.randn(n, 1, generator=generator)).float()
    return [
        (x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, n, batch_size)
    ]


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 1))


def test_fit_reduces_validation_loss():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    history = fit(
        model,
        nn.functional.mse_loss,
        optimizer,
        make_batches(256, 32, seed=1),
        make_batches(64, 32, seed=2),
        epochs=30,
    )
    assert len(history["train"]) == len(history["val"]) == 30
    assert history["val"][-1] < history["val"][0] / 5


def test_eval_step_is_deterministic_and_leaves_no_gradients():
    model = make_model()
    eval_step = make_eval_step(model, nn.functional.mse_loss)
    (x, y), *_ = make_batches(32, 32, seed=3)
    # Dropout is active in train mode, so two evaluations would differ.
    assert eval_step(x, y) == eval_step(x, y)
    assert all(p.grad is None for p in model.parameters())
    assert not model.training


def test_eval_step_runs_without_autograd():
    model = make_model()

    def spying_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled(), "evaluation must not build a graph"
        assert not yhat.requires_grad
        return nn.functional.mse_loss(yhat, y)

    eval_step = make_eval_step(model, spying_loss)
    (x, y), *_ = make_batches(32, 32, seed=3)
    eval_step(x, y)


def test_train_step_puts_the_model_in_train_mode():
    model = make_model()
    model.eval()
    train_step = make_train_step(
        model, nn.functional.mse_loss, torch.optim.SGD(model.parameters(), lr=0.1)
    )
    (x, y), *_ = make_batches(32, 32, seed=3)
    train_step(x, y)
    assert model.training
