# Solution -- 08.07 torch.func: grad, vmap and functional_call
import torch
from torch.func import functional_call, grad, vmap


def per_sample_gradients(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    """{name: (B, *param.shape)}: the loss gradient for each example alone."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(
        p: dict[str, torch.Tensor], xi: torch.Tensor, yi: torch.Tensor
    ) -> torch.Tensor:
        pred = functional_call(model, p, (xi.unsqueeze(0),))
        return ((pred.squeeze(0) - yi) ** 2).mean()

    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)


def _by_loop(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    out: dict[str, list[torch.Tensor]] = {k: [] for k, _ in model.named_parameters()}
    for xi, yi in zip(x, y, strict=True):
        model.zero_grad()
        ((model(xi.unsqueeze(0)).squeeze(0) - yi) ** 2).mean().backward()
        for k, p in model.named_parameters():
            out[k].append(p.grad.detach().clone())
    return {k: torch.stack(v) for k, v in out.items()}


def test_per_sample_gradients_match_a_loop():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    got = per_sample_gradients(model, x, y)
    want = _by_loop(model, x, y)
    assert got.keys() == want.keys()
    for k in want:
        assert got[k].shape == (8, *dict(model.named_parameters())[k].shape)
        torch.testing.assert_close(got[k], want[k])


def test_examples_get_different_gradients():
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    g = per_sample_gradients(model, x, y)["weight"]
    assert not torch.allclose(g[0], g[1])
