# =============================================================================
#  08.03 -- no_grad and inference_mode
# =============================================================================
#
#  Autograd records every operation on a tensor that requires grad. Two
#  kinds of code must NOT be recorded:
#
#   * The parameter update. `p -= lr * p.grad` is an in-place operation on
#     a leaf that requires grad; outside `torch.no_grad()` autograd refuses
#     it ("a leaf Variable that requires grad is being used in an in-place
#     operation"). Inside, it is just arithmetic. Godoy, ch. 1 "no_grad":
#     "It allows us to perform regular Python operations on tensors without
#     affecting PyTorch's computation graph."
#   * Evaluation. Recording the graph for a forward pass whose result will
#     never be differentiated costs memory (every intermediate is kept) and
#     time. `torch.inference_mode()` is `no_grad` plus: the tensors it
#     produces are marked as inference tensors and cannot later be used in
#     autograd at all, which lets PyTorch skip more bookkeeping. Use it for
#     evaluation and serving, `no_grad` where the results might still feed
#     into a graph later.
#
#  `.detach()` is the third tool: a view of the same data, cut out of the
#  graph.
#
#  TASK
#    Make `sgd_update` legal, and make `predict` run without recording.
#
#  RUN IT
#    ./npt test 08_03
#
# =============================================================================

import torch


def sgd_update(params: list[torch.Tensor], lr: float) -> None:
    """In-place gradient step on every parameter that has a gradient."""
    # TODO: autograd refuses this in-place update of a leaf. Wrap it.
    for p in params:
        if p.grad is not None:
            p -= lr * p.grad


def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A forward pass for evaluation: no graph, no grad, no surprises."""
    # TODO: this builds a graph nobody will use. Run it in inference mode.
    return model(x)


def test_sgd_update_moves_the_parameters():
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    (p**2).sum().backward()
    sgd_update([p], lr=0.5)
    torch.testing.assert_close(p.detach(), torch.tensor([0.0, 0.0]))
    assert p.is_leaf
    assert p.requires_grad


def test_sgd_update_skips_parameters_without_grad():
    p = torch.tensor([1.0], requires_grad=True)
    sgd_update([p], lr=0.5)
    torch.testing.assert_close(p.detach(), torch.tensor([1.0]))


def test_predict_records_nothing():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    out = predict(model, torch.randn(4, 3))
    assert out.shape == (4, 1)
    assert not out.requires_grad
    assert out.grad_fn is None
    assert out.is_inference()
