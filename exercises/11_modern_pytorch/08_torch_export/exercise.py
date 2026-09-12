# =============================================================================
#  11.08 -- torch.export
# =============================================================================
#
#  Sooner or later a model has to leave Python: to be served from C++, run
#  through TensorRT or ONNX, or handed to someone who must not need your
#  training environment. For a decade the answer was TorchScript --
#  `torch.jit.trace` and `torch.jit.script` -- and a great deal of code still
#  says so. Do not write new code that way: since PyTorch 2.x TorchScript is
#  in maintenance, and in 2.14 every `torch.jit.*` entry point emits a
#  deprecation warning pointing at the replacement
#  (https://docs.pytorch.org/docs/2.14/export.html):
#
#      from torch.export import export, Dim
#      program = export(model, (example_input,),
#                       dynamic_shapes={"x": {0: Dim("batch")}})
#      program.module()(x)                     # runnable, no Python model needed
#      torch.export.save(program, "model.pt2")
#      program = torch.export.load("model.pt2")
#
#  `export` uses the same tracer as `torch.compile` (11.02), so it sees the
#  whole program as ONE graph of ATen operators with the weights baked in as
#  inputs, and it fails loudly on anything it cannot capture -- which is
#  what you want from a serialisation step, and is what `jit.trace` never
#  did (it silently baked in whichever branch the example input took).
#
#  Shapes are the part to get right. By default every dimension of the
#  example input is a CONSTANT of the exported program: export with a batch
#  of 4 and the program rejects a batch of 11. `dynamic_shapes` marks the
#  dimensions that may vary, by argument name and position, with a `Dim`
#  object; the dimensions you leave out stay fixed and are checked at run
#  time, which is a feature -- a wrong feature count is an error, not a
#  silent broadcast.
#
#  TASK
#    Replace the TorchScript trace with `torch.export.export`, make the batch
#    dimension dynamic, and implement the save/load round trip.
#
#  RUN IT
#    ./npt test 11_08
#
# =============================================================================

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.export import Dim, ExportedProgram, export  # noqa: F401


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
    # TODO: TorchScript is deprecated, and a trace specialises to the example
    # shape. Use torch.export.export with dynamic_shapes marking dimension 0
    # of `x` as Dim("batch").
    return torch.jit.trace(model, example)  # type: ignore[return-value]


def round_trip(program: ExportedProgram, path: Path) -> ExportedProgram:
    """Save the program to disk and load it back."""
    # TODO: torch.export.save / torch.export.load, not torch.save (an
    # ExportedProgram is not a state_dict).
    torch.save(program, path)
    return torch.load(path, weights_only=False)


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
