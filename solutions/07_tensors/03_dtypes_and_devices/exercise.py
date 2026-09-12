# Solution -- 07.03 dtypes and devices
import torch


def running_mean_buffer(x: torch.Tensor) -> torch.Tensor:
    """A zeroed per-feature accumulator matching `x`'s dtype and device."""
    return torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)


def pick_device() -> torch.device:
    """The accelerator if there is one, else the CPU."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def promoted_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    """The dtype `a + b` would have, without computing it."""
    return torch.result_type(a, b)


def test_buffer_follows_dtype():
    x = torch.ones(4, 3, dtype=torch.float64)
    buf = running_mean_buffer(x)
    assert buf.shape == (3,)
    assert buf.dtype == torch.float64
    # It must be usable in the accumulation without a dtype error.
    buf += x.mean(dim=0)
    torch.testing.assert_close(buf, torch.ones(3, dtype=torch.float64))


def test_buffer_follows_device():
    x = torch.empty(2, 5, device="meta")
    buf = running_mean_buffer(x)
    assert buf.device.type == "meta"
    assert buf.shape == (5,)


def test_pick_device_returns_a_device_object():
    dev = pick_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda", "mps", "xpu"}
    # Whatever it is, a tensor can be made there.
    assert torch.zeros(1, device=dev).device.type == dev.type


def test_promotion_rules():
    f32 = torch.zeros(2, dtype=torch.float32)
    f64 = torch.zeros(2, dtype=torch.float64)
    i64 = torch.zeros(2, dtype=torch.int64)
    assert promoted_dtype(f32, f64) == torch.float64
    assert promoted_dtype(f64, f32) == torch.float64
    assert promoted_dtype(i64, f32) == torch.float32
    assert promoted_dtype(f32, torch.tensor(1.0, dtype=torch.float64)) == torch.float32
