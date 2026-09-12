# Solution -- 12.01 Shape errors

import torch
from torch import nn

Spec = tuple[int | str | None, ...]


def expect_shape(
    t: torch.Tensor, spec: Spec, bound: dict[str, int] | None = None
) -> dict[str, int]:
    """Check `t.shape` against `spec` and return the named dimensions.

    An int must match exactly, None matches anything, and a string names a
    dimension: its first occurrence binds the name, later ones must agree.
    """
    bound = dict(bound or {})
    if t.ndim != len(spec):
        raise ValueError(f"expected {len(spec)} dims {spec}, got shape {tuple(t.shape)}")
    for axis, (want, got) in enumerate(zip(spec, t.shape, strict=True)):
        if want is None:
            continue
        if isinstance(want, str):
            if want in bound and bound[want] != got:
                raise ValueError(
                    f"dim {axis} is {want}={got} but {want} was {bound[want]} earlier; "
                    f"shape {tuple(t.shape)} against {spec}"
                )
            bound[want] = got
        elif want != got:
            raise ValueError(
                f"dim {axis}: expected {want}, got {got}; "
                f"shape {tuple(t.shape)} against {spec}"
            )
    return bound


class TokenClassifier(nn.Module):
    """Classify every token of a (B, T, C) sequence into `classes`."""

    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.mix = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.head = nn.Linear(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The channel count is KNOWN here, so say so: a (B, C, T) input with
        # C != T is rejected at the door with "dim 2: expected 8, got 5".
        dims = expect_shape(x, ("B", "T", self.mix.in_channels))
        dims["C"] = self.mix.in_channels
        # Conv1d wants (B, C, T); Linear wants the feature axis last. Every
        # transpose is written out and checked, so a wrong layout fails HERE
        # with a message naming the axis, not three layers later.
        h = self.mix(x.transpose(1, 2))
        expect_shape(h, ("B", "C", "T"), dims)
        h = h.transpose(1, 2)
        expect_shape(h, ("B", "T", "C"), dims)
        out = self.head(h)
        expect_shape(out, ("B", "T", self.head.out_features), dims)
        return out


def test_expect_shape_binds_names():
    assert expect_shape(torch.zeros(2, 5, 7), ("B", "T", 7)) == {"B": 2, "T": 5}


def test_expect_shape_rejects_wrong_rank_and_wrong_size():
    try:
        expect_shape(torch.zeros(2, 5), ("B", "T", "C"))
    except ValueError as e:
        assert "3 dims" in str(e)
    else:
        raise AssertionError("wrong rank accepted")
    try:
        expect_shape(torch.zeros(2, 5, 7), ("B", "T", 8))
    except ValueError as e:
        assert "dim 2" in str(e)
    else:
        raise AssertionError("wrong size accepted")


def test_expect_shape_enforces_agreement_between_uses_of_a_name():
    dims = expect_shape(torch.zeros(2, 5, 7), ("B", "T", "C"))
    try:
        expect_shape(torch.zeros(2, 6, 7), ("B", "T", "C"), dims)
    except ValueError as e:
        assert "T" in str(e)
    else:
        raise AssertionError("a changed T was accepted")
    # And within one tensor.
    try:
        expect_shape(torch.zeros(3, 4), ("N", "N"))
    except ValueError:
        pass
    else:
        raise AssertionError("a non-square tensor matched (N, N)")


def test_token_classifier_maps_b_t_c_to_b_t_classes():
    torch.manual_seed(0)
    model = TokenClassifier(channels=8, classes=4)
    out = model(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)


def test_token_classifier_rejects_channels_first_input_with_a_clear_message():
    torch.manual_seed(0)
    model = TokenClassifier(channels=8, classes=4)
    try:
        model(torch.randn(2, 8, 5))  # (B, C, T): the layout of the previous layer
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("a (B, C, T) input was accepted")
