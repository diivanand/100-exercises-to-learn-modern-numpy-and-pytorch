# Solution -- 09.03 Sequential and initialisation
import math
from collections import OrderedDict

import torch
from torch import nn


def build_mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    """fc1 -> ReLU -> fc2, with the layers named so they can be addressed."""
    # An OrderedDict gives the children names. With the positional form they
    # are "0", "1", "2", which is what you see in every state_dict key and
    # every error message for the rest of the model's life.
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(n_in, n_hidden)),
                ("act", nn.ReLU()),
                ("fc2", nn.Linear(n_hidden, n_out)),
            ]
        )
    )


def init_weights(module: nn.Module) -> None:
    """Kaiming-uniform weights and zero biases, for Linear layers only."""
    # `apply` visits every submodule, including the ReLU, which has no
    # weight. Dispatch on the type rather than assuming.
    if isinstance(module, nn.Linear):
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
