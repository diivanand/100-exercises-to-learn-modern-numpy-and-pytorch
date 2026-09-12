# =============================================================================
#  09.09 -- Initialisation and activations
# =============================================================================
#
#  A deep network is a product of many matrices. Multiply a signal by ten
#  matrices whose entries are a little too small and it vanishes; a little
#  too large and it explodes. Either way the gradients that come back through
#  the same matrices do the same, and training does not start. Kneusel
#  (*Practical Deep Learning*, ch. 6) shows the effect empirically:
#  "the performance of a traditional neural network is strongly influenced
#  by the type of random initialization used".
#
#  The scale that keeps the variance steady depends on the activation. For a
#  layer with `fan_in` inputs:
#
#      linear / tanh / sigmoid    Var(w) = 1 / fan_in         (Xavier / Glorot)
#      ReLU                       Var(w) = 2 / fan_in         (Kaiming / He)
#
#  The 2 is because ReLU zeroes half of its inputs, halving the variance
#  at every layer; the initialisation doubles it back. PyTorch spells these
#  `nn.init.xavier_normal_` and `nn.init.kaiming_normal_(w, nonlinearity=...)`.
#  Biases start at zero.
#
#  The related failure is the DEAD ReLU: a unit whose pre-activation is
#  negative for every input in the data produces zero, receives zero
#  gradient, and never recovers. A large negative bias, or weights so large
#  that a few units dominate, kills them in bulk. Godoy ch. 4, "Activation
#  Functions", surveys the alternatives (leaky ReLU, PReLU, and today GELU and
#  SiLU) that keep a small gradient alive on the negative side.
#
#  TASK
#    Initialise the layers so that the activation scale survives ten layers
#    and almost no unit is dead.
#
#  RUN IT
#    ./npt test 09_09
#
# =============================================================================

import torch
from torch import nn


def make_deep_mlp(width: int, depth: int) -> nn.Sequential:
    """`depth` Linear layers of `width`, ReLU between them, Kaiming-initialised."""
    layers: list[nn.Module] = []
    for i in range(depth):
        linear = nn.Linear(width, width)
        # TODO: "small random numbers" is not a scheme. Ten layers of it and
        # the signal is gone; the bias makes matters worse.
        nn.init.normal_(linear.weight, std=0.01)
        nn.init.constant_(linear.bias, -0.1)
        layers.append(linear)
        if i < depth - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def activation_stds(model: nn.Sequential, x: torch.Tensor) -> list[float]:
    """Standard deviation of the output of every Linear layer, in order."""
    stds = []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                stds.append(x.std().item())
    return stds


def dead_fraction(model: nn.Sequential, x: torch.Tensor) -> float:
    """Fraction of hidden units that are zero for EVERY input (dead ReLUs)."""
    with torch.no_grad():
        dead = []
        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                dead.append((x == 0).all(dim=0))
    return torch.cat(dead).float().mean().item()


def test_signal_scale_is_preserved_through_ten_layers():
    torch.manual_seed(0)
    model = make_deep_mlp(width=256, depth=10)
    x = torch.randn(1024, 256)
    stds = activation_stds(model, x)
    assert len(stds) == 10
    # With a bad init these drift by orders of magnitude; 0.3 to 3 is generous.
    assert all(0.3 < s < 3.0 for s in stds), stds


def test_few_units_are_dead():
    torch.manual_seed(0)
    model = make_deep_mlp(width=256, depth=10)
    x = torch.randn(1024, 256)
    assert dead_fraction(model, x) < 0.05


def test_biases_start_at_zero():
    model = make_deep_mlp(width=8, depth=3)
    for layer in model:
        if isinstance(layer, nn.Linear):
            assert torch.all(layer.bias == 0)


def test_weight_std_is_kaiming():
    torch.manual_seed(0)
    model = make_deep_mlp(width=512, depth=2)
    expected = (2.0 / 512) ** 0.5
    assert abs(model[0].weight.std().item() - expected) < 0.1 * expected
