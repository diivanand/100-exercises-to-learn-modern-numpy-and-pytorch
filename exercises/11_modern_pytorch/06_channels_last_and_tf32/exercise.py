# =============================================================================
#  11.06 -- channels_last and TF32
# =============================================================================
#
#  Two switches that cost one line each and are worth a large fraction of the
#  4090's convolution and matmul throughput. Neither changes what the code
#  computes, within rounding, and both are invisible in the tensor shapes,
#  which is why they are easy to get wrong.
#
#  CHANNELS_LAST. PyTorch's image tensors are NCHW (Godoy ch. 4, "Shape
#  (NCHW vs NHWC)"): the logical order is batch, channel, height, width. The
#  Tensor Core convolution kernels in cuDNN want the channels of one pixel
#  next to each other in memory -- NHWC. You do NOT permute the tensor: that
#  would change its logical shape and break every layer after it. You change
#  the MEMORY FORMAT, which is a stride pattern:
#
#      x = x.contiguous(memory_format=torch.channels_last)   # still (N, C, H, W)
#      model = model.to(memory_format=torch.channels_last)   # weights too
#
#  `x.shape` is unchanged, `x.stride()[1] == 1`, `x.is_contiguous()` is now
#  False and `x.is_contiguous(memory_format=torch.channels_last)` is True.
#  Convolutions propagate the format, so converting the input and the model
#  once is enough. (https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
#
#  TF32. Since Ampere, Tensor Cores can run a float32 matmul with 10-bit
#  mantissas internally (TensorFloat-32), about 8x faster than true float32.
#  PyTorch defaults to "highest" precision, i.e. off. Turn it on with
#
#      torch.set_float32_matmul_precision("high")
#
#  ("high" = TF32 when available, "medium" = bfloat16 where allowed). The
#  getter `torch.get_float32_matmul_precision()` tells you what is in force,
#  and a training script should log it, because a run that is 3x slower than
#  a colleague's is more often this than anything else. The loss of precision
#  is irrelevant for training and can matter for linear algebra; 04.02's
#  ill-conditioned solve is exactly the kind of computation to keep at
#  "highest". `torch.backends.cudnn.benchmark = True` is the third switch:
#  cuDNN tries every algorithm for each new input shape and caches the best,
#  which is right for fixed-shape training and wrong for variable shapes.
#
#  The tests run on the CPU, where channels_last is honoured (the strides
#  change, the values do not) and the precision setting is recorded but has
#  no effect. On the 4090 all three make a measurable difference; 11.05 has
#  the tool to measure it with.
#
#  TASK
#    Make `to_channels_last` change the memory format rather than the shape,
#    and make `enable_fast_float32_matmul` set and report "high".
#
#  RUN IT
#    ./npt test 11_06
#
# =============================================================================

import torch
from torch import nn


def to_channels_last(
    model: nn.Module, images: torch.Tensor
) -> tuple[nn.Module, torch.Tensor]:
    """Put a conv net and its NCHW input into the channels_last memory format."""
    # TODO: this permutes the logical dimensions to NHWC, which is a different
    # tensor as far as Conv2d is concerned. Use the memory_format argument of
    # .contiguous() for the images and of .to() for the model instead.
    images = images.permute(0, 2, 3, 1).contiguous()
    return model, images


def enable_fast_float32_matmul() -> str:
    """Allow TensorFloat-32 for float32 matmuls and convolutions."""
    # TODO: "highest" is the default and means TF32 off. Set "high", turn on
    # cudnn.benchmark, and return what the getter reports.
    torch.set_float32_matmul_precision("highest")
    return "high"


def test_channels_last_keeps_the_shape_and_changes_the_strides():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    images = torch.randn(2, 3, 16, 16)
    model_cl, images_cl = to_channels_last(model, images)
    assert images_cl.shape == (2, 3, 16, 16)
    assert images_cl.is_contiguous(memory_format=torch.channels_last)
    assert not images_cl.is_contiguous()
    # Channel stride 1: neighbouring channels of one pixel are adjacent.
    assert images_cl.stride()[1] == 1
    assert model_cl[0].weight.is_contiguous(memory_format=torch.channels_last)


def test_channels_last_propagates_through_the_convolution():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    images = torch.randn(2, 3, 16, 16)
    reference = model(images)
    model_cl, images_cl = to_channels_last(model, images)
    out = model_cl(images_cl)
    assert out.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(out, reference)


def test_fast_matmul_setting_is_reported():
    before = torch.get_float32_matmul_precision()
    try:
        assert enable_fast_float32_matmul() == "high"
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.set_float32_matmul_precision(before)
        torch.backends.cudnn.benchmark = False
