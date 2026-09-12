# Solution -- 08.04 The graph, leaves and .item()
import torch


def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error, scaled by 1 / (1 + max abs error) as a tensor."""
    err = pred - target
    scale = 1.0 / (1.0 + err.abs().max())
    return scale * (err**2).mean()


def hidden_grad(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(sum(relu(h))) / dh where h = x @ w, an intermediate, not a leaf."""
    h = x @ w
    h.retain_grad()
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
