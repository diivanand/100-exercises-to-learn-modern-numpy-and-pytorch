# =============================================================================
#  07.08 -- squeeze, unsqueeze and permute
# =============================================================================
#
#  Deep learning code is mostly shape plumbing. The verbs:
#
#      x.unsqueeze(d)     insert a size-1 dimension at position d
#      x.squeeze(d)       remove dimension d IF it has size 1
#      x.squeeze()        remove EVERY size-1 dimension -- the trap
#      x.permute(...)     reorder dimensions (a view, like transpose)
#      x.flatten(start)   merge dimensions from `start` onward
#      x.movedim(a, b)    move one dimension
#
#  PyTorch's image layout is NCHW: (batch, channels, height, width) (Godoy,
#  ch. 4 "Shape (NCHW vs NHWC)"). A single greyscale image has shape
#  (1, 1, H, W) as a batch. Bare `.squeeze()` on it returns (H, W): the
#  batch dimension is gone, and so is the channel dimension, and the next
#  layer raises a shape error that mentions neither. Always name the
#  dimension you squeeze.
#
#  TASK
#    Fix `remove_batch_dim`, and write `to_nhwc` and `flatten_features`
#    without hard-coding sizes.
#
#  RUN IT
#    ./npt test 07_08
#
# =============================================================================

import torch


def add_batch_dim(image: torch.Tensor) -> torch.Tensor:
    """(C, H, W) -> (1, C, H, W)."""
    return image.unsqueeze(0)


def remove_batch_dim(batch: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) -> (C, H, W), whatever C, H and W are."""
    # TODO: a bare squeeze also removes a channel dimension of 1.
    return batch.squeeze()


def to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) -> (N, H, W, C), as a view."""
    # TODO: transpose swaps two dimensions; this needs a permutation.
    return x.transpose(1, 3)


def flatten_features(x: torch.Tensor) -> torch.Tensor:
    """(N, ...) -> (N, everything else), for a classifier head."""
    # TODO: this flattens the batch dimension away as well.
    return x.flatten()


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
