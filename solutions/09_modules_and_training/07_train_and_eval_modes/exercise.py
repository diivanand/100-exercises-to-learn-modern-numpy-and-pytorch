# Solution -- 09.07 Train and eval modes
import torch
from torch import nn


def make_model(n_in: int = 4, n_hidden: int = 32) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.BatchNorm1d(n_hidden),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(n_hidden, 1),
    )


def train_one_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float = 0.01
) -> float:
    """One SGD step. Dropout is on and BatchNorm uses batch statistics."""
    model.train()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad
            p.grad = None
    return loss.item()


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predictions for deployment: deterministic, and touching no state."""
    # eval() switches Dropout off and makes BatchNorm use its running
    # statistics instead of the batch's. It also stops BatchNorm UPDATING
    # those statistics, which is why predicting must never happen in train
    # mode: every prediction call would nudge the model.
    model.eval()
    with torch.inference_mode():
        return model(x)


def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = 3.0 + 2.0 * torch.randn(n, 4, generator=generator)
    y = x.sum(dim=1, keepdim=True)
    return x, y


def test_predictions_are_deterministic():
    model = make_model()
    x, _ = make_data(16, seed=1)
    torch.testing.assert_close(predict(model, x), predict(model, x))


def test_predicting_does_not_change_running_statistics():
    model = make_model()
    x, y = make_data(64, seed=1)
    train_one_step(model, x, y)
    bn = model[1]
    before = bn.running_mean.clone()
    batches_before = bn.num_batches_tracked.clone()
    predict(model, x[:8])
    predict(model, x[8:24])
    torch.testing.assert_close(bn.running_mean, before)
    assert bn.num_batches_tracked == batches_before


def test_training_does_change_running_statistics():
    model = make_model()
    x, y = make_data(64, seed=1)
    bn = model[1]
    assert torch.all(bn.running_mean == 0)
    train_one_step(model, x, y)
    assert not torch.all(bn.running_mean == 0)
    assert bn.num_batches_tracked == 1


def test_prediction_uses_running_statistics_not_batch_statistics():
    model = make_model()
    x, y = make_data(64, seed=1)
    for _ in range(5):
        train_one_step(model, x, y)
    # A single point has no batch statistics to speak of. In train mode
    # BatchNorm1d refuses it outright; in eval mode it is just a point.
    single = predict(model, x[:1])
    from_batch = predict(model, x)[:1]
    torch.testing.assert_close(single, from_batch)
