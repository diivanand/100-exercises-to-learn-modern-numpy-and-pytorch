# Solution -- 08.03 no_grad and inference_mode
import torch


def sgd_update(params: list[torch.Tensor], lr: float) -> None:
    """In-place gradient step on every parameter that has a gradient."""
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= lr * p.grad


def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A forward pass for evaluation: no graph, no grad, no surprises."""
    with torch.inference_mode():
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
