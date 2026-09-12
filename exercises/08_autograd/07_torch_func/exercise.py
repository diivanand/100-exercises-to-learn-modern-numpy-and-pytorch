# =============================================================================
#  08.07 -- torch.func: grad, vmap and functional_call
# =============================================================================
#
#  `torch.func` (the successor of functorch, in core since PyTorch 2.0) is
#  autograd as function transforms, in the JAX style:
#
#      torch.func.grad(f)(x)          the gradient of scalar f at x
#      torch.func.vmap(f)(batch)      f applied per row, vectorised
#      torch.func.functional_call(model, params, (x,))
#                                     run a module with SUBSTITUTED parameters
#
#  The reason to care is per-sample gradients. Differential privacy, data
#  attribution and some optimisers need the gradient of the loss for EACH
#  example separately. `loss.backward()` on a batch gives you the gradient
#  of the mean; a Python loop over examples is correct but slow. With
#  `grad` for one example and `vmap` over the batch dimension, PyTorch
#  vectorises the loop into one batched computation.
#
#  `functional_call` is the piece that lets a module's parameters be
#  ordinary tensors the transforms can see.
#
#  TASK
#    Make `per_sample_gradients` return one gradient per example, not the
#    batch gradient repeated.
#
#  RUN IT
#    ./npt test 08_07
#
# =============================================================================

import torch
from torch.func import functional_call, grad


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

    # TODO: this differentiates the loss of the WHOLE batch and repeats it
    # B times. Wrap grad(loss_one) in vmap over the batch dimension of x
    # and y (in_dims=(None, 0, 0): the parameters are shared). Import vmap
    # from torch.func alongside grad.
    def loss_batch(p: dict[str, torch.Tensor]) -> torch.Tensor:
        return ((functional_call(model, p, (x,)) - y) ** 2).mean()

    g = grad(loss_batch)(params)
    return {k: v.unsqueeze(0).expand(x.shape[0], *v.shape) for k, v in g.items()}


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
