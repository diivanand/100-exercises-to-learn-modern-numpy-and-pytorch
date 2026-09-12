# Solution -- 07.01 Creating tensors
import torch


def features_from_ints(values: list[int]) -> torch.Tensor:
    """Turn integer measurements into a float32 feature vector."""
    return torch.tensor(values, dtype=torch.float32)


def scalar_and_vector(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `value` both as a 0-d tensor and as a length-1 vector."""
    return torch.tensor(value), torch.tensor([value])


def time_grid(start: float, stop: float, n: int) -> torch.Tensor:
    """`n` equally spaced float32 samples, including both endpoints."""
    return torch.linspace(start, stop, n)


def test_features_are_float32():
    f = features_from_ints([3, 1, 4])
    assert f.dtype == torch.float32
    torch.testing.assert_close(f, torch.tensor([3.0, 1.0, 4.0]))


def test_scalar_and_vector_shapes():
    s, v = scalar_and_vector(2.5)
    assert s.shape == ()
    assert s.ndim == 0
    assert v.shape == (1,)
    assert s.item() == 2.5
    assert v.item() == 2.5


def test_time_grid_has_both_endpoints():
    g = time_grid(0.0, 1.0, 11)
    assert g.shape == (11,)
    assert g.dtype == torch.float32
    assert g[0] == 0.0
    assert g[-1] == 1.0
    torch.testing.assert_close(g[5], torch.tensor(0.5))


def test_time_grid_length_is_exact_for_awkward_steps():
    # 0.1 is not representable in binary; a step-based construction can end
    # up one element short or long. Asking for n points cannot.
    assert time_grid(0.0, 0.3, 4).shape == (4,)
    assert time_grid(-1.0, 1.0, 7).shape == (7,)
