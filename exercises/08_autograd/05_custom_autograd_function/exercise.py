# =============================================================================
#  08.05 -- A custom autograd Function
# =============================================================================
#
#  When an operation is not made of differentiable pieces -- it calls into
#  C, or its naive derivative is numerically bad -- you write the forward
#  and backward yourself as a `torch.autograd.Function`:
#
#      class MyOp(torch.autograd.Function):
#          @staticmethod
#          def forward(ctx, x):
#              ctx.save_for_backward(x)      # what backward will need
#              return ...
#          @staticmethod
#          def backward(ctx, grad_output):
#              (x,) = ctx.saved_tensors
#              return grad_output * ...      # one gradient per forward input
#
#  and call it with `MyOp.apply(x)`. `backward` receives dL/d(output) and
#  must return dL/d(input) by the chain rule -- one entry per argument of
#  forward, `None` for arguments that are not tensors.
#
#  `torch.autograd.gradcheck(fn, inputs)` compares your backward against
#  finite differences. Run it in float64: in float32 the finite-difference
#  estimate itself is too noisy to trust. This is the unit test every
#  custom Function should have.
#
#  The example is softplus, log(1 + e^x), whose textbook form overflows
#  for x > 88 in float32; `torch.logaddexp(x, 0)` is the stable spelling
#  and its derivative is the sigmoid.
#
#  TASK
#    Write `backward` so that gradcheck passes.
#
#  RUN IT
#    ./npt test 08_05
#
# =============================================================================

import torch


class StableSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.logaddexp(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO: this treats softplus as the identity. Its derivative is the
        # sigmoid of the saved input; apply the chain rule to grad_output.
        return grad_output


def softplus(x: torch.Tensor) -> torch.Tensor:
    return StableSoftplus.apply(x)


def test_forward_matches_the_library_and_is_stable():
    x = torch.tensor([-100.0, 0.0, 100.0])
    torch.testing.assert_close(softplus(x), torch.nn.functional.softplus(x))
    assert torch.isfinite(softplus(x)).all()


def test_backward_matches_the_library():
    x = torch.linspace(-4, 4, 9, requires_grad=True)
    softplus(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.sigmoid(x.detach()))


def test_gradcheck_in_float64():
    x = torch.randn(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(softplus, (x,))
