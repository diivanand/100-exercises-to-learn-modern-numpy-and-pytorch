# Solution -- 07.08 squeeze, unsqueeze and permute
import torch


def add_batch_dim(image: torch.Tensor) -> torch.Tensor:
    """(C, H, W) -> (1, C, H, W)."""
    return image.unsqueeze(0)


def remove_batch_dim(batch: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) -> (C, H, W), whatever C, H and W are."""
    return batch.squeeze(0)


def to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) -> (N, H, W, C), as a view."""
    return x.permute(0, 2, 3, 1)


def flatten_features(x: torch.Tensor) -> torch.Tensor:
    """(N, ...) -> (N, everything else), for a classifier head."""
    return x.flatten(start_dim=1)


def test_batch_round_trip_with_one_channel():
    image = torch.zeros(1, 8, 6)
    batch = add_batch_dim(image)
    assert batch.shape == (1, 1, 8, 6)
    assert remove_batch_dim(batch).shape == (1, 8, 6)


def test_to_nhwc_is_a_view_with_the_right_order():
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    y = to_nhwc(x)
    assert y.shape == (2, 4, 5, 3)
    assert y[1, 2, 3, 0] == x[1, 0, 2, 3]
    assert y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


def test_flatten_features_keeps_the_batch():
    x = torch.zeros(4, 3, 2, 2)
    assert flatten_features(x).shape == (4, 12)
    assert flatten_features(torch.zeros(1, 7)).shape == (1, 7)
