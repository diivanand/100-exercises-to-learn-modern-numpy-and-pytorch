# =============================================================================
#  07.02 -- The NumPy bridge
# =============================================================================
#
#  NumPy and PyTorch share memory across the bridge, and that is both the
#  feature and the trap.
#
#   * `torch.from_numpy(arr)` and `torch.as_tensor(arr)` return a tensor that
#     SHARES the array's memory. No copy, and a write on either side is seen
#     by the other (Godoy, ch. 1 "Loading Data, Devices, and CUDA": "if you
#     modify the original Numpy array, you're modifying the corresponding
#     PyTorch tensor too, and vice-versa").
#   * `torch.tensor(arr)` always COPIES.
#   * `tensor.numpy()` shares memory the other way, and only works for a CPU
#     tensor that does not require grad: you must `.detach()` (chapter 08)
#     and `.cpu()` first. `t.detach().cpu().numpy()` is the idiom.
#
#  The dtype trap: NumPy's default float is float64, PyTorch's is float32.
#  An array created with `rng.normal(...)` is float64, so a tensor made from
#  it is float64, and feeding that to an `nn.Linear` whose weights are
#  float32 raises "mat1 and mat2 must have the same dtype". Convert at the
#  boundary, once, with `.float()` or `dtype=`.
#
#  TASK
#    Make `share` share memory, make `model_input` produce float32, and make
#    `to_numpy` work on any CPU tensor, including one that requires grad.
#
#  RUN IT
#    ./npt test 07_02
#
# =============================================================================

import numpy as np
import torch


def share(arr: np.ndarray) -> torch.Tensor:
    """A tensor over the same memory as `arr`, with no copy."""
    # TODO: torch.tensor copies. Use the function that wraps the array.
    return torch.tensor(arr)


def model_input(arr: np.ndarray) -> torch.Tensor:
    """A float32 tensor ready to go through a float32 model."""
    # TODO: the array is float64, so this tensor is float64, and a float32
    # model will refuse it. Convert at the boundary.
    return torch.as_tensor(arr)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Back to NumPy, whatever the tensor's autograd state."""
    # TODO: this raises for a tensor that requires grad. Detach it first
    # (and move it to the CPU: a GPU tensor cannot be viewed by NumPy).
    return t.numpy()


def test_share_does_not_copy():
    arr = np.arange(5, dtype=np.float32)
    t = share(arr)
    arr[2] = 100.0
    assert t[2] == 100.0
    t[3] = -1.0
    assert arr[3] == -1.0


def test_model_input_is_float32_and_usable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    assert x.dtype == np.float64
    t = model_input(x)
    assert t.dtype == torch.float32
    layer = torch.nn.Linear(3, 2)
    out = layer(t)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(t.numpy(), x.astype(np.float32))


def test_to_numpy_handles_grad_tensors():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    y = t * 2
    arr = to_numpy(y)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [2.0, 4.0])
    # The original still takes part in autograd.
    assert y.requires_grad
