# =============================================================================
#  12.01 -- Shape errors
# =============================================================================
#
#  Most bugs in tensor code are shape bugs, and the worst of them do not
#  raise. Broadcasting (07.07) happily turns a (B, 1) minus a (B,) into a
#  (B, B); a transposed batch still goes through a Linear if the last axis
#  happens to have the right size; `.view` reinterprets rather than
#  rearranges. The error, when there is one, comes from a layer deep in the
#  model with a message about mat1 and mat2 and no mention of your variable.
#
#  The habit that fixes this is cheap: name your axes in comments, and CHECK
#  them at the boundaries, with a helper that says which axis was wrong:
#
#      dims = expect_shape(x, ("B", "T", "C"))       # binds B, T, C
#      expect_shape(h, ("B", "T", 64), dims)          # T must agree with x
#
#  A string names a dimension and must be consistent everywhere it appears;
#  an int must match exactly; None is "anything". The helper returns the
#  bindings so later checks can enforce agreement. This is what the einops
#  and jaxtyping libraries do with more machinery; a twenty-line function
#  gets you most of the value with no dependency. Effective Python, 3rd ed.,
#  Item 81 makes the general case: assert internal assumptions where they
#  are cheap to check and expensive to violate silently.
#
#  The model below is a small example of the trap. It takes (B, T, C), runs
#  a Conv1d -- which wants (B, C, T) -- and a Linear -- which wants C last.
#  Given a (B, C, T) input by mistake, a version without checks either fails
#  inside Conv1d with a message about "expected input[2, 8, 5] to have 5
#  channels" or, when C == T, produces garbage with no error at all.
#
#  TASK
#    Complete `expect_shape` so that names are bound and enforced, and check
#    the layout at each stage of `TokenClassifier.forward`.
#
#  RUN IT
#    ./npt test 12_01
#
# =============================================================================

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
            # TODO: a name that is already bound must agree with `got`;
            # raise a ValueError that mentions the name if it does not.
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
        # TODO: nothing here checks the layout. Check the input against
        # ("B", "T", self.mix.in_channels) -- the channel count is known, so
        # use the number -- transpose to (B, C, T) for the convolution, back
        # to (B, T, C) for the head, and check each stage against the
        # bindings.
        h = self.mix(x)
        return self.head(h)


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
