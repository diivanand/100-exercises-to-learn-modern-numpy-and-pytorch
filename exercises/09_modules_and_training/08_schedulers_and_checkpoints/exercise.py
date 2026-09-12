# =============================================================================
#  09.08 -- Schedulers and checkpoints
# =============================================================================
#
#  A LEARNING-RATE SCHEDULER changes the optimiser's learning rate as training
#  goes on (Godoy ch. 6, "Learning Rate Schedulers"). It wraps the optimiser,
#  and it has a `step()` of its own, called once per epoch (for the epoch-
#  based ones such as StepLR) AFTER the optimiser's `step()`:
#
#      scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
#      for epoch in range(epochs):
#          for x, y in batches:
#              ...; optimizer.step(); optimizer.zero_grad()
#          scheduler.step()
#
#  A CHECKPOINT is what lets a run survive a crash, a pre-emption or a laptop
#  lid. Godoy's rule (ch. 2, "Saving and Loading Models", "Model State"): the
#  state of a training run is not the model. It is
#
#      model.state_dict()        the parameters and buffers
#      optimizer.state_dict()    momentum buffers, Adam's moment estimates
#      scheduler.state_dict()    which epoch the schedule thinks it is
#      the epoch counter, and anything else you would need to carry on
#
#  Save only the model and you can DEPLOY it ("Deploying / Making
#  Predictions"). Resume training from it and the first step after loading is
#  taken with fresh, empty momentum and the learning rate back at its
#  starting value: a different run, silently.
#
#  LOADING. `torch.load(path, weights_only=True)` restricts the unpickler to
#  tensors, containers and primitives. It has been the default since 2.6,
#  because a checkpoint is a pickle, and a pickle can run code. Write it
#  explicitly anyway, so a reader knows you thought about it.
#
#  TASK
#    Save and restore the whole training state, so that a run stopped at
#    epoch 3 and resumed matches a run that was never stopped.
#
#  RUN IT
#    ./npt test 09_08
#
# =============================================================================

from pathlib import Path
from typing import Any

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))


def make_optimizer_and_scheduler(
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # Multiply the learning rate by 0.5 every two epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return optimizer, scheduler


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: list[Batch],
    epochs: int,
) -> list[float]:
    """Train for `epochs` epochs and return the learning rate used in each."""
    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        for x, y in batches:
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # The scheduler steps once per epoch, after the optimizer has stepped.
        scheduler.step()
    return lrs


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
) -> None:
    # TODO: this saves a model you can deploy, not a run you can resume. The
    # optimiser's momentum, the scheduler's position and the epoch are lost.
    torch.save({"model": model.state_dict()}, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Restore everything and return the epoch to resume from."""
    # TODO: restore the optimiser and scheduler too, load with
    # weights_only=True, and return the saved epoch rather than assuming 0.
    checkpoint: dict[str, Any] = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    return 0


def make_batches(seed: int = 0) -> list[Batch]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(128, 2, generator=generator)
    y = torch.sin(x[:, :1]) + x[:, 1:] ** 2
    return [(x[i : i + 32], y[i : i + 32]) for i in range(0, 128, 32)]


def test_learning_rate_halves_every_two_epochs():
    model = make_model()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    lrs = train(model, optimizer, scheduler, make_batches(), epochs=6)
    assert lrs == [0.1, 0.1, 0.05, 0.05, 0.025, 0.025]


def test_resumed_training_matches_uninterrupted_training(tmp_path: Path):
    batches = make_batches()

    # Six epochs straight through.
    reference = make_model()
    optimizer, scheduler = make_optimizer_and_scheduler(reference)
    train(reference, optimizer, scheduler, batches, epochs=6)

    # Three epochs, save, forget everything, load, three more.
    model = make_model()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    train(model, optimizer, scheduler, batches, epochs=3)
    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, epoch=3)
    del model, optimizer, scheduler

    resumed = make_model()
    optimizer, scheduler = make_optimizer_and_scheduler(resumed)
    start = load_checkpoint(tmp_path / "ckpt.pt", resumed, optimizer, scheduler)
    assert start == 3
    lrs = train(resumed, optimizer, scheduler, batches, epochs=6 - start)
    assert lrs == [0.05, 0.025, 0.025]

    for p_ref, p_res in zip(reference.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(p_ref, p_res)


def test_checkpoint_loads_with_weights_only(tmp_path: Path):
    model = make_model()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, epoch=1)
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=True)
    assert set(checkpoint) == {"model", "optimizer", "scheduler", "epoch"}
