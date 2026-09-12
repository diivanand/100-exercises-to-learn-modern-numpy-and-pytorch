# Solution -- 11.08 torch.export

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.export import Dim, ExportedProgram, export


class Classifier(nn.Module):
    def __init__(self, features: int = 8, classes: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(features, 16), nn.GELU(), nn.Linear(16, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).softmax(dim=-1)


def export_classifier(model: nn.Module, example: torch.Tensor) -> ExportedProgram:
    """Export `model` with a dynamic batch dimension."""
    # Without dynamic_shapes the program is specialised to the example's
    # batch size and refuses any other. Dim("batch") makes dimension 0 of the
    # first argument symbolic; every other dimension stays static.
    batch = Dim("batch")
    return export(model, (example,), dynamic_shapes={"x": {0: batch}})


def round_trip(program: ExportedProgram, path: Path) -> ExportedProgram:
    """Save the program to disk and load it back."""
    torch.export.save(program, path)
    return torch.export.load(path)


def test_exported_program_runs_and_agrees_with_eager():
    torch.manual_seed(0)
    model = Classifier().eval()
    example = torch.randn(4, 8)
    program = export_classifier(model, example)
    assert isinstance(program, ExportedProgram)
    torch.testing.assert_close(program.module()(example), model(example))


def test_batch_dimension_is_dynamic():
    torch.manual_seed(0)
    model = Classifier().eval()
    program = export_classifier(model, torch.randn(4, 8))
    other = torch.randn(11, 8)
    torch.testing.assert_close(program.module()(other), model(other))


def test_feature_dimension_is_still_checked():
    torch.manual_seed(0)
    model = Classifier().eval()
    program = export_classifier(model, torch.randn(4, 8))
    # The exported program guards the static dimensions and refuses the call.
    with pytest.raises((AssertionError, RuntimeError), match="size"):
        program.module()(torch.randn(4, 9))


def test_round_trip_through_a_file(tmp_path):
    torch.manual_seed(0)
    model = Classifier().eval()
    example = torch.randn(5, 8)
    program = export_classifier(model, example)
    loaded = round_trip(program, tmp_path / "classifier.pt2")
    assert isinstance(loaded, ExportedProgram)
    torch.testing.assert_close(loaded.module()(example), model(example))
