# =============================================================================
#  09.03 -- Sequential and initialisation
# =============================================================================
#
#  `nn.Sequential` is a Module that calls its children in order. It is the
#  right tool when the model IS a pipeline and there is nothing to say in
#  `forward` (Godoy ch. 1, "Sequential Models"). It has two forms:
#
#      nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
#      nn.Sequential(OrderedDict([("fc1", ...), ("act", ...), ("fc2", ...)]))
#
#  With the first, the children are named "0", "1", "2". Those names are what
#  you will read in every `state_dict` key, every checkpoint, every hook and
#  every error message, for as long as the model exists. Name them.
#
#  INITIALISATION. Each layer initialises itself when constructed (Linear
#  uses a Kaiming-uniform variant). To apply your own scheme, write a
#  function that initialises one module, and let `model.apply(fn)` walk the
#  tree and call it on every submodule -- including the ReLU, which has no
#  weight at all, and the Sequential itself. So the function dispatches on
#  the module's type and does nothing for the rest. 09.09 covers WHY the
#  scheme matters; here it is only the mechanics.
#
#  TASK
#    Name the layers `fc1`, `act`, `fc2`, and make `init_weights` initialise
#    only the Linear layers.
#
#  RUN IT
#    ./npt test 09_03
#
# =============================================================================

import math

import torch
from torch import nn


def build_mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    """fc1 -> ReLU -> fc2, with the layers named so they can be addressed."""
    # TODO: the children of this Sequential are called "0", "1" and "2".
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))


def init_weights(module: nn.Module) -> None:
    """Kaiming-uniform weights and zero biases, for Linear layers only."""
    # TODO: `apply` calls this on EVERY submodule. Not all of them have a
    # weight.
    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    nn.init.zeros_(module.bias)


def test_children_are_named():
    model = build_mlp(4, 8, 2)
    assert list(dict(model.named_children())) == ["fc1", "act", "fc2"]
    assert isinstance(model.fc1, nn.Linear) and model.fc1.out_features == 8
    assert set(model.state_dict()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def test_forward_shape():
    model = build_mlp(4, 8, 2)
    assert model(torch.zeros(5, 4)).shape == (5, 2)


def test_init_touches_every_linear_and_nothing_else():
    torch.manual_seed(0)
    model = build_mlp(512, 512, 10)
    model.apply(init_weights)
    for name in ("fc1", "fc2"):
        layer = getattr(model, name)
        assert torch.all(layer.bias == 0)
    # Kaiming uniform with gain sqrt(2): bound = sqrt(6 / fan_in), so the
    # standard deviation is sqrt(2 / fan_in).
    expected_std = math.sqrt(2.0 / 512)
    assert abs(model.fc1.weight.std().item() - expected_std) < 0.1 * expected_std


def test_init_is_seeded():
    torch.manual_seed(1)
    a = build_mlp(4, 8, 2)
    a.apply(init_weights)
    torch.manual_seed(1)
    b = build_mlp(4, 8, 2)
    b.apply(init_weights)
    torch.testing.assert_close(a.fc1.weight, b.fc1.weight)
