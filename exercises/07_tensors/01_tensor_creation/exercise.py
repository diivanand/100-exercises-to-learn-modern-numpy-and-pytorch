# =============================================================================
#  07.01 -- Creating tensors
# =============================================================================
#
#  A tensor is NumPy's ndarray with two additions: it can live on an
#  accelerator, and it can remember how it was computed (chapter 08). The
#  creation functions look like NumPy's, and most of the surprises are in the
#  places where they quietly differ.
#
#   * `torch.tensor(data)` COPIES `data` and INFERS the dtype from it: a list
#     of Python ints becomes int64, a list of floats becomes float32 -- not
#     float64 as NumPy would choose. `torch.get_default_dtype()` is float32.
#     Almost every model wants float32, so a tensor built from integer data
#     needs `dtype=torch.float32` said out loud.
#
#   * `torch.tensor(3.0)` is a 0-d tensor (shape `()`); `torch.tensor([3.0])`
#     has shape `(1,)`. They print differently and broadcast the same, so
#     the confusion usually shows up later as an off-by-one in `ndim`.
#
#   * `torch.Tensor(3)` (capital T, the legacy constructor) makes an
#     UNINITIALISED float tensor of shape `(3,)`. It is not "the number 3".
#     Do not use it; it exists for backwards compatibility.
#
#   * `torch.arange(a, b, step)` excludes `b`, like Python's range;
#     `torch.linspace(a, b, n)` includes it. `torch.zeros`, `ones`, `full`,
#     `eye`, `rand`, `randn` mirror NumPy; `torch.empty` is uninitialised.
#
#  Godoy, ch. 1 "Tensor", walks through scalars, vectors, matrices and shapes
#  with `torch.as_tensor`; the type-inference rule above is the reason his
#  code calls `.float()` on everything.
#
#  TASK
#    Fix the three constructors so that the dtypes and shapes match the tests.
#
#  RUN IT
#    ./npt test 07_01
#
# =============================================================================

import torch


def features_from_ints(values: list[int]) -> torch.Tensor:
    """Turn integer measurements into a float32 feature vector."""
    # TODO: this infers int64 from the Python ints. Say the dtype you want.
    return torch.tensor(values)


def scalar_and_vector(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `value` both as a 0-d tensor and as a length-1 vector."""
    # TODO: `torch.Tensor(value)` is the legacy constructor: it allocates an
    # uninitialised tensor of size `value`, it does not hold the number.
    return torch.Tensor(value), torch.tensor([value])


def time_grid(start: float, stop: float, n: int) -> torch.Tensor:
    """`n` equally spaced float32 samples, including both endpoints."""
    # TODO: arange excludes the stop value and its length depends on float
    # rounding of the step. linspace is the tool for "n points, both ends".
    step = (stop - start) / (n - 1)
    return torch.arange(start, stop, step)


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
