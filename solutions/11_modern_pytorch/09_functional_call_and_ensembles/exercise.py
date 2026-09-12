# Solution -- 11.09 functional_call and ensembles

import copy

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


def make_ensemble(n: int, features: int = 8, classes: int = 3) -> list[nn.Module]:
    torch.manual_seed(0)
    return [
        nn.Sequential(nn.Linear(features, 16), nn.Tanh(), nn.Linear(16, classes))
        for _ in range(n)
    ]


def ensemble_forward(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Run every model on the same `x` in one batched call. Returns (n, B, C)."""
    # stack_module_state gives {name: (n, *shape)} for parameters and buffers.
    params, buffers = stack_module_state(models)
    # A "meta" copy has the structure but no storage: it is the template that
    # functional_call fills in with the stacked tensors, one slice per model.
    base = copy.deepcopy(models[0]).to("meta")

    def call_one(
        p: dict[str, torch.Tensor], b: dict[str, torch.Tensor], inputs: torch.Tensor
    ):
        return functional_call(base, (p, b), (inputs,))

    # in_dims=(0, 0, None): map over the model axis of params and buffers,
    # broadcast x to every model.
    return vmap(call_one, in_dims=(0, 0, None))(params, buffers, x)


def ensemble_mean(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Average the ensemble's softmax outputs. Returns (B, C)."""
    return ensemble_forward(models, x).softmax(dim=-1).mean(dim=0)


def test_ensemble_forward_matches_running_each_model():
    models = make_ensemble(4)
    x = torch.randn(5, 8)
    out = ensemble_forward(models, x)
    assert out.shape == (4, 5, 3)
    reference = torch.stack([m(x) for m in models])
    torch.testing.assert_close(out, reference)


def test_models_are_left_untouched():
    models = make_ensemble(3)
    before = [copy.deepcopy(m.state_dict()) for m in models]
    ensemble_forward(models, torch.randn(2, 8))
    for m, state in zip(models, before, strict=True):
        for name, value in m.state_dict().items():
            assert value.device.type != "meta"
            torch.testing.assert_close(value, state[name])


def test_ensemble_mean_is_a_distribution():
    models = make_ensemble(4)
    probs = ensemble_mean(models, torch.randn(6, 8))
    assert probs.shape == (6, 3)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(6))


def test_ensemble_forward_is_a_single_batched_call():
    # Under vmap, the model's forward runs ONCE on batched tensors whose
    # visible shape is the per-model shape. A Python loop would run it once
    # per model. Count forward calls with a hook on the first layer.
    models = make_ensemble(4)
    calls: list[int] = []
    handles = [m[0].register_forward_hook(lambda *_: calls.append(1)) for m in models]
    try:
        ensemble_forward(models, torch.randn(2, 8))
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) <= 1, f"the ensemble ran {len(calls)} separate forward passes"
