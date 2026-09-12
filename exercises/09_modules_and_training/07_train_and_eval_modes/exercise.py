# =============================================================================
#  09.07 -- Train and eval modes
# =============================================================================
#
#  Every Module has a boolean, `training`, toggled by `model.train()` and
#  `model.eval()` and propagated to every submodule. Most layers ignore it.
#  Two families do not, and they are in nearly every real model:
#
#   * DROPOUT (Godoy ch. 6, "Dropout"). In train mode it zeroes each
#     activation with probability p and scales the survivors by 1/(1-p). In
#     eval mode it is the identity. A model predicting in train mode gives a
#     different answer for the same input every time.
#
#   * BATCHNORM (Godoy ch. 7, "Batch Normalization"). In train mode it
#     normalises each feature by the statistics of the CURRENT BATCH, and
#     updates a running mean and variance as a side effect. In eval mode it
#     uses the running statistics and updates nothing. Predicting in train
#     mode therefore does two wrong things at once: the answer depends on
#     what else was in the batch (a batch of one is refused outright), and
#     the model's buffers drift every time you call it.
#
#  Godoy puts it in capitals: "After loading the model, DO NOT FORGET to SET
#  THE MODE: checkpointing: model.train(); deploying / making predictions:
#  model.eval()" (ch. 2, "Setting the Model's Mode").
#
#  Note what `eval()` does NOT do: it does not switch autograd off. That is a
#  separate switch, `torch.inference_mode()` (09.06, 08.03), and a prediction
#  function wants both.
#
#  TASK
#    Make `predict` put the model in eval mode and run without autograd.
#
#  RUN IT
#    ./npt test 09_07
#
# =============================================================================

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
    # TODO: the model is still in whatever mode the last caller left it in,
    # and autograd is recording.
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
