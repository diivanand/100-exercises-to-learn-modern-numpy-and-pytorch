# =============================================================================
#  11.09 -- functional_call and ensembles
# =============================================================================
#
#  An `nn.Module` carries its parameters with it, which is convenient right
#  up to the moment you want to run the SAME architecture with DIFFERENT
#  parameters: an ensemble of models on one input, a model-agnostic
#  meta-learning step, per-sample gradients (08.07). Then the pairing of
#  "this code" with "these weights" is in the way.
#
#  `torch.func` (https://docs.pytorch.org/docs/2.14/func.api.html; the old
#  `functorch` package is this API under its previous name) separates them:
#
#      functional_call(module, (params, buffers), (x,))
#
#  runs `module.forward` with the given dicts in place of the module's own
#  tensors, without mutating the module. Combine it with `vmap`, which turns
#  a function of one example into a function of a batch by adding a mapped
#  dimension, and an ensemble becomes one call:
#
#      params, buffers = stack_module_state(models)   # {name: (n, *shape)}
#      base = copy.deepcopy(models[0]).to("meta")     # structure, no storage
#      f = lambda p, b, x: functional_call(base, (p, b), (x,))
#      vmap(f, in_dims=(0, 0, None))(params, buffers, x)   # -> (n, B, C)
#
#  `in_dims` says which argument has the mapped dimension and where: 0 for
#  the stacked tensors, None for `x`, which every model sees whole. The
#  "meta" device is the trick that makes the template free: a meta tensor
#  has a shape and dtype and no memory, so the copy costs nothing and cannot
#  accidentally be used for computation.
#
#  Why not a Python loop? For four small models on a CPU the loop is fine.
#  On the 4090 with 32 models, the loop launches 32 sets of kernels each
#  doing a sliver of work, while vmap launches ONE set of batched matmuls
#  that fill the GPU. It is the same reason a batch beats a loop over
#  samples, applied one level up. The last test checks that the forward pass
#  really ran once.
#
#  TASK
#    Rewrite `ensemble_forward` as one vmap over stacked parameters.
#
#  RUN IT
#    ./npt test 11_09
#
# =============================================================================

import copy

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap  # noqa: F401


def make_ensemble(n: int, features: int = 8, classes: int = 3) -> list[nn.Module]:
    torch.manual_seed(0)
    return [
        nn.Sequential(nn.Linear(features, 16), nn.Tanh(), nn.Linear(16, classes))
        for _ in range(n)
    ]


def ensemble_forward(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Run every model on the same `x` in one batched call. Returns (n, B, C)."""
    # TODO: this is n separate forward passes. Stack the parameters and
    # buffers with stack_module_state, make a "meta" copy of one model as
    # the template, and vmap a functional_call over the stacked state with
    # in_dims=(0, 0, None).
    return torch.stack([model(x) for model in models])


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
