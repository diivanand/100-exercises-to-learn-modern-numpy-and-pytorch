# Solution -- 09.09 Initialisation and activations
import torch
from torch import nn


def make_deep_mlp(width: int, depth: int) -> nn.Sequential:
    """`depth` Linear layers of `width`, ReLU between them, Kaiming-initialised."""
    layers: list[nn.Module] = []
    for i in range(depth):
        linear = nn.Linear(width, width)
        # Kaiming (He) normal: std = sqrt(2 / fan_in). The 2 compensates for
        # ReLU discarding half of each layer's variance, so the signal keeps
        # its scale from layer to layer instead of shrinking or exploding.
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        nn.init.zeros_(linear.bias)
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
