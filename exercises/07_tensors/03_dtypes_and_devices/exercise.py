# =============================================================================
#  07.03 -- dtypes and devices
# =============================================================================
#
#  Every tensor has a dtype and a device, and the code that creates a tensor
#  in the middle of a computation has to say which. The default is
#  float32-on-CPU, which is right until the first time it is not: a helper
#  called with a float64 tensor, or with a tensor on the GPU, then fails with
#  "Expected all tensors to be on the same device" from somewhere deep in a
#  matmul.
#
#  The rule: new tensors follow the tensor they will be combined with.
#  `torch.zeros(shape, dtype=x.dtype, device=x.device)`, or the shortcut
#  `torch.zeros_like(x)`, or `x.new_zeros(shape)`.
#
#  Type promotion follows NumPy's NEP 50 rules: a float32 tensor and a
#  Python float give float32; a float32 tensor and a float64 TENSOR give
#  float64. `torch.result_type(a, b)` tells you without computing anything.
#
#  Choosing a device in 2026: Godoy's `'cuda' if torch.cuda.is_available()
#  else 'cpu'` predates Apple Silicon (`mps`) and Intel GPUs (`xpu`). The
#  device-agnostic spelling since PyTorch 2.6 is
#
#      dev = (torch.accelerator.current_accelerator()
#             if torch.accelerator.is_available() else torch.device("cpu"))
#
#  which yields `mps` on this course's authoring laptop, `cuda` on the RTX
#  4090 box, and `cpu` in CI, with no branches in the model code. 11.01
#  builds a whole training script on that.
#
#  The tests use the `meta` device: a tensor with a shape and dtype but no
#  storage, which makes "follows its input's device" checkable on any
#  machine.
#
#  TASK
#    Make `running_mean_buffer` follow its input, and implement
#    `pick_device` as above.
#
#  RUN IT
#    ./npt test 07_03
#
# =============================================================================

import torch


def running_mean_buffer(x: torch.Tensor) -> torch.Tensor:
    """A zeroed per-feature accumulator matching `x`'s dtype and device."""
    # TODO: this is float32 on the CPU whatever `x` is. Follow the input.
    return torch.zeros(x.shape[-1])


def pick_device() -> torch.device:
    """The accelerator if there is one, else the CPU."""
    # TODO: this only knows about CUDA. Use torch.accelerator, which also
    # covers Apple's mps and Intel's xpu, and return a torch.device.
    return "cuda" if torch.cuda.is_available() else "cpu"


def promoted_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    """The dtype `a + b` would have, without computing it."""
    # TODO: ask torch.result_type instead of guessing that the first wins.
    return a.dtype


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
