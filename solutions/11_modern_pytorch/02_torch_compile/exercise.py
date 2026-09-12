# Solution -- 11.02 torch.compile

from collections.abc import Callable

import torch


def gated_update(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale `x` where its mean is positive; otherwise shift it down by one."""
    # `torch.where` keeps the decision inside the tensor world, so the whole
    # function is one graph. The Python `if` in the starter forced Dynamo to
    # stop, evaluate the condition eagerly, and start a second graph.
    positive = x.mean() > 0
    return torch.where(positive, x * scale, x - 1.0)


def compile_gated_update(backend: str | Callable = "inductor") -> Callable:
    """Return a compiled `gated_update` that must capture a single graph."""
    # fullgraph=True turns a silent graph break into an error at compile time.
    # For a small function that is the right default: you find out now, not
    # from a profiler three weeks later.
    return torch.compile(gated_update, backend=backend, fullgraph=True)


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
