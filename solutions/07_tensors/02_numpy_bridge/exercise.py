# Solution -- 07.02 The NumPy bridge
import numpy as np
import torch


def share(arr: np.ndarray) -> torch.Tensor:
    """A tensor over the same memory as `arr`, with no copy."""
    return torch.from_numpy(arr)


def model_input(arr: np.ndarray) -> torch.Tensor:
    """A float32 tensor ready to go through a float32 model."""
    # as_tensor may share memory when dtypes already match; the cast makes a
    # copy only when it has to.
    return torch.as_tensor(arr, dtype=torch.float32)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Back to NumPy, whatever the tensor's autograd state."""
    return t.detach().cpu().numpy()


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
