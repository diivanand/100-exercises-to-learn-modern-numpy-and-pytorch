# =============================================================================
#  11.02 -- torch.compile
# =============================================================================
#
#  Eager PyTorch runs one operator at a time: Python calls a kernel, waits for
#  the result, calls the next. `torch.compile(fn)` (PyTorch 2.0+) records the
#  sequence of operators the first time `fn` runs, hands the graph to a
#  backend, and returns a callable that runs the backend's fused code. On the
#  4090 the default backend, Inductor, generates Triton kernels and typically
#  buys 1.5-2x on training loops; `mode="reduce-overhead"` adds CUDA graphs on
#  top, which removes launch overhead for small batches
#  (https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).
#
#  The recording is done by Dynamo, which reads the Python bytecode. It can
#  trace through most of what you write, but not through a decision that
#  depends on a tensor VALUE:
#
#      if x.mean() > 0:        # Dynamo cannot know which branch without
#          return x * scale    # running the graph, so it stops here,
#      return x - 1.0          # runs it, and starts a second graph.
#
#  That is a GRAPH BREAK. Nothing is wrong, exactly; you just get two small
#  graphs and an eager step between them, and most of the benefit is gone.
#  Two tools make breaks visible:
#
#      torch._dynamo.explain(fn)(*args)     # .graph_break_count, .break_reasons
#      torch.compile(fn, fullgraph=True)    # refuses to compile if it must break
#
#  The fix is to keep the decision in the tensor world: `torch.where(cond, a, b)`
#  is one operator that computes both branches and selects.
#
#  Two more things you will meet: `dynamic=None` (the default) recompiles on a
#  new shape and then marks that dimension dynamic, and everything after
#  `.item()` or a Python `print` is a break as well.
#
#  The tests here do not use Inductor. Inductor needs a C++ compiler for CPU
#  code and Triton for the GPU, and neither belongs in a unit test. They use
#  a tiny custom backend that counts how many graphs it received, which is
#  also the cleanest way to see what Dynamo captured. On the 4090, try:
#
#      compiled = torch.compile(model, mode="reduce-overhead")
#
#  and look at the timeline with `nsys profile python train.py`.
#
#  TASK
#    Remove the graph break from `gated_update`, and make `compile_gated_update`
#    insist on a single graph.
#
#  RUN IT
#    ./npt test 11_02
#
# =============================================================================

from collections.abc import Callable

import torch


def gated_update(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale `x` where its mean is positive; otherwise shift it down by one."""
    # TODO: this `if` depends on a tensor value, so Dynamo has to break the
    # graph here. Express the choice with torch.where instead.
    if x.mean() > 0:
        return x * scale
    return x - 1.0


def compile_gated_update(backend: str | Callable = "inductor") -> Callable:
    """Return a compiled `gated_update` that must capture a single graph."""
    # TODO: ask for fullgraph=True, so that a graph break is an error rather
    # than a silent slowdown.
    return torch.compile(gated_update, backend=backend)


def test_gated_update_matches_the_specification():
    x = torch.tensor([1.0, -0.5, 2.0])
    torch.testing.assert_close(gated_update(x, 3.0), torch.tensor([3.0, -1.5, 6.0]))
    y = torch.tensor([-1.0, -2.0])
    torch.testing.assert_close(gated_update(y, 3.0), torch.tensor([-2.0, -3.0]))


def test_there_are_no_graph_breaks():
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(gated_update)(torch.randn(8), 2.0)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1


def test_compiled_function_runs_through_the_backend_and_agrees_with_eager():
    torch._dynamo.reset()
    captured: list[int] = []

    def counting_backend(gm: torch.fx.GraphModule, example_inputs):
        captured.append(len(gm.graph.nodes))
        return gm.forward

    compiled = compile_gated_update(backend=counting_backend)
    torch.manual_seed(0)
    x = torch.randn(16)
    torch.testing.assert_close(compiled(x, 2.5), gated_update(x, 2.5))
    torch.testing.assert_close(compiled(-x.abs(), 2.5), gated_update(-x.abs(), 2.5))
    assert len(captured) >= 1, "the backend was never called: nothing was compiled"
