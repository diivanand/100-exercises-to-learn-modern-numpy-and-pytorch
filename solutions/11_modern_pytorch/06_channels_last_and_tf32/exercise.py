# Solution -- 11.06 channels_last and TF32

import torch
from torch import nn


def to_channels_last(
    model: nn.Module, images: torch.Tensor
) -> tuple[nn.Module, torch.Tensor]:
    """Put a conv net and its NCHW input into the channels_last memory format."""
    # The logical shape stays (N, C, H, W); only the strides change, so that C
    # is the fastest-varying dimension in memory. Nothing else in the program
    # needs to know.
    model = model.to(memory_format=torch.channels_last)
    images = images.contiguous(memory_format=torch.channels_last)
    return model, images


def enable_fast_float32_matmul() -> str:
    """Allow TensorFloat-32 for float32 matmuls and convolutions."""
    # "high" = TF32 where the hardware has it (Ampere and later), "highest" =
    # true float32. The getter reports what is in force so a script can log it.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return torch.get_float32_matmul_precision()


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
