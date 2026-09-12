# Solution -- 09.08 Schedulers and checkpoints
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
    # A checkpoint that can RESUME training holds everything with state: the
    # parameters, the optimizer (momentum buffers, Adam moments), the
    # scheduler (which epoch it thinks it is), and the epoch counter. Saving
    # only model.state_dict() gives you a model you can deploy, not one you can
    # keep training as if nothing happened.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Restore everything and return the epoch to resume from."""
    # weights_only=True has been the default since 2.6: the unpickler accepts
    # tensors, dicts, and primitives, and refuses arbitrary code. Say it
    # explicitly anyway; a checkpoint is data you may not have written.
    checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"]


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
