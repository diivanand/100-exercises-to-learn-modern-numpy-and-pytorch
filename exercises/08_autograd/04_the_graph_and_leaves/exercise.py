# =============================================================================
#  08.04 -- The graph, leaves and .item()
# =============================================================================
#
#  Every tensor that came out of an autograd-tracked operation carries a
#  `grad_fn` (AddBackward0, MulBackward0, ...), the node that knows how to
#  push a gradient back to its inputs. Godoy, ch. 1 "Dynamic Computation
#  Graph", draws these as grey boxes between the blue leaves (parameters)
#  and the green output. Three consequences:
#
#   * `.item()`, `float()`, `.numpy()` and `.tolist()` leave the graph. A
#     Python float has no grad_fn, so anything computed from it downstream
#     cannot be differentiated: `backward()` raises "element 0 of tensors
#     does not require grad and does not have a grad_fn". Keep the loss a
#     tensor until after `backward()`; call `.item()` only to LOG it.
#   * Only leaves get `.grad`. An intermediate you want to inspect needs
#     `.retain_grad()` before `backward()`.
#   * The graph is freed by `backward()`. A second `backward()` through the
#     same graph raises "Trying to backward through the graph a second
#     time" unless the first was called with `retain_graph=True` -- and if
#     you find yourself wanting that, you usually wanted 08.06 instead.
#
#  TASK
#    Keep `weighted_loss` differentiable, and make `hidden_grad` return the
#    gradient of the intermediate.
#
#  RUN IT
#    ./npt test 08_04
#
# =============================================================================

import torch


def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error, scaled by 1 / (1 + max abs error) as a tensor."""
    err = pred - target
    # TODO: .item() turns the scale into a Python float and cuts it out of
    # the graph. Keep it a tensor; the scale should be differentiated too.
    scale = 1.0 / (1.0 + err.abs().max().item())
    return scale * (err**2).mean()


def hidden_grad(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(sum(relu(h))) / dh where h = x @ w, an intermediate, not a leaf."""
    h = x @ w
    # TODO: h is not a leaf, so its .grad is None after backward unless you
    # ask for it to be retained before calling backward.
    torch.relu(h).sum().backward()
    return h.grad


def test_weighted_loss_is_differentiable_through_the_scale():
    pred = torch.tensor([1.0, 3.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    loss = weighted_loss(pred, target)
    assert loss.grad_fn is not None
    loss.backward()
    # scale = 1/(1+3) = 1/4, mse = (1 + 9)/2 = 5, loss = 1.25
    torch.testing.assert_close(loss.detach(), torch.tensor(1.25))
    # d/dpred[1] includes the derivative of the scale through max():
    # d(mse)/dp1 * scale + mse * d(scale)/dp1 = 3 * 0.25 + 5 * (-1/16)
    torch.testing.assert_close(pred.grad[1], torch.tensor(3 * 0.25 - 5 / 16))
    torch.testing.assert_close(pred.grad[0], torch.tensor(1 * 0.25))


def test_hidden_grad_is_the_relu_mask():
    w = torch.tensor([[1.0, -1.0]], requires_grad=True)
    x = torch.tensor([[2.0], [-3.0]])
    g = hidden_grad(w, x)
    assert g is not None
    # h = [[2, -2], [-3, 3]]; d relu / dh = (h > 0)
    torch.testing.assert_close(g, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
