# =============================================================================
#  10.05 -- Transforms and normalisation
# =============================================================================
#
#  Standardising the inputs -- subtracting a mean and dividing by a standard
#  deviation, per feature -- is the transform that nearly every model wants
#  and nearly every tutorial does wrong. Godoy ch. 4 ("Normalize Transform")
#  explains the name confusion: Torchvision calls it Normalize, scikit-learn
#  calls it StandardScaler, and `torch.nn.functional.normalize` is something
#  else entirely (unit vectors, not unit variance).
#
#  The mistake is not the arithmetic. It is WHERE the statistics come from.
#  The validation split stands in for data the model will meet after
#  deployment. If the mean and standard deviation are computed over the
#  training and validation data together, then the validation set has
#  influenced the pipeline the model is scored with. That is LEAKAGE. It
#  makes the validation score optimistic in a way that is invisible until the
#  model meets real data, whose statistics were not available at
#  fitting time.
#
#  The scikit-learn shape of the fix is worth copying even without
#  scikit-learn: an object with `fit(x_train)` that learns the statistics
#  and `transform(x)` that applies them, so the same object is used for both
#  splits and it is impossible to compute the statistics twice.
#
#  TASK
#    Make `prepare` fit the standardiser on the training split alone.
#
#  RUN IT
#    ./npt test 10_05
#
# =============================================================================

import torch


class Standardiser:
    """Per-feature (x - mean) / std, with statistics learned from one split."""

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "Standardiser":
        self.mean = x.mean(dim=0)
        # Guard against a constant feature; dividing by zero makes NaNs.
        self.std = x.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() before transform()")
        return (x - self.mean) / self.std


def prepare(
    x_train: torch.Tensor, x_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Standardiser]:
    """Standardise both splits using statistics from the training split only."""
    # TODO: the statistics are computed over both splits. The validation data
    # has leaked into the preprocessing.
    everything = torch.cat([x_train, x_val])
    scaler = Standardiser().fit(everything)
    return scaler.transform(x_train), scaler.transform(x_val), scaler


def make_splits() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x_train = 5.0 + 3.0 * torch.randn(200, 4, generator=generator)
    # Validation data from a shifted distribution, as happens in practice.
    x_val = 8.0 + 3.0 * torch.randn(50, 4, generator=generator)
    return x_train, x_val


def test_training_split_is_standard_after_transform():
    x_train, x_val = make_splits()
    z_train, _, _ = prepare(x_train, x_val)
    torch.testing.assert_close(z_train.mean(dim=0), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(z_train.std(dim=0), torch.ones(4), atol=1e-5, rtol=0)


def test_statistics_come_from_the_training_split_only():
    x_train, x_val = make_splits()
    _, z_val, scaler = prepare(x_train, x_val)
    torch.testing.assert_close(scaler.mean, x_train.mean(dim=0))
    torch.testing.assert_close(scaler.std, x_train.std(dim=0))
    # The validation split is shifted, and must still look shifted.
    assert z_val.mean().item() > 0.5


def test_same_transform_for_both_splits():
    x_train, x_val = make_splits()
    z_train, z_val, scaler = prepare(x_train, x_val)
    torch.testing.assert_close(z_val, scaler.transform(x_val))
    torch.testing.assert_close(z_train, scaler.transform(x_train))


def test_transform_before_fit_is_an_error():
    try:
        Standardiser().transform(torch.zeros(2, 2))
    except RuntimeError:
        return
    raise AssertionError("transform() before fit() must raise")
