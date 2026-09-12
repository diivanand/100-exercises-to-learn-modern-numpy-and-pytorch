# =============================================================================
#  04.02 -- Solve, do not invert
# =============================================================================
#
#  The textbook writes the solution of A x = b as x = A^-1 b, so the textbook
#  translation into code is `np.linalg.inv(a) @ b`. Do not write that. It is
#  slower (an inverse is n solves, then a product) and, more importantly, it
#  is less accurate: forming A^-1 rounds every entry of the inverse, and the
#  product then mixes those errors into every component of x. `np.linalg.solve`
#  factorises A once (LU with partial pivoting) and substitutes, which keeps
#  the error at the level of the condition number, not the condition number
#  squared.
#
#  Johansson, *Numerical Python* ch. 5, "Square Systems", derives the bound
#  that matters: the relative error in x is at most cond(A) times the
#  relative error in b, where cond(A) = ||A^-1|| ||A||. With float64's sixteen
#  digits, cond(A) ~ 1e12 leaves you four. np.linalg.cond computes it, and
#  np.linalg.matrix_rank tells you when it is infinite (Johansson's example
#  is a singular 2x2; the tests below use a Vandermonde matrix, which is
#  the classic ill-conditioned one).
#
#  The other thing the textbook does not tell you: `solve` takes a stack of
#  right-hand sides. `np.linalg.solve(a, B)` with B of shape (n, k) solves k
#  systems with ONE factorisation.
#
#  Kneusel, *Math for Deep Learning* ch. 6, "Inverses", is the linear-algebra
#  background if you want it.
#
#  TASK
#    Make `solve_system` solve without forming an inverse (one of the tests
#    forbids np.linalg.inv outright), and finish `condition_report`.
#
#  RUN IT
#    ./npt test 04_02
#
# =============================================================================

import numpy as np


def solve_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a @ x = b. `b` may be a vector (n,) or a stack of columns (n, k)."""
    # TODO: this is the textbook formula and the wrong code. Use
    # np.linalg.solve; it already handles a stack of right-hand sides.
    return np.linalg.inv(a) @ b


def condition_report(a: np.ndarray) -> tuple[float, bool]:
    """The 2-norm condition number of `a`, and whether it is safe to solve with.

    "Safe" here means cond(a) < 1e12: with float64's ~16 significant digits,
    that leaves about four digits of accuracy in the solution.
    """
    # TODO: use np.linalg.cond. Note that cond of a singular matrix is inf,
    # which compares False against everything, so spell the check out.
    return 0.0, True


def test_solves_a_well_conditioned_system():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    np.testing.assert_allclose(solve_system(a, b), [2.0, 3.0])


def test_solves_many_right_hand_sides_at_once():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 6)) + 6 * np.eye(6)
    b = rng.standard_normal((6, 4))
    x = solve_system(a, b)
    assert x.shape == (6, 4)
    np.testing.assert_allclose(a @ x, b, atol=1e-12)


def test_keeps_precision_on_an_ill_conditioned_system():
    # A 12x12 Vandermonde matrix has a condition number around 1e9. Solving via
    # the explicit inverse leaves a residual around 1e-9; an LU solve leaves
    # one around 1e-16. The bound below sits comfortably between the two.
    n = 12
    a = np.vander(np.linspace(0.0, 1.0, n), n)
    rng = np.random.default_rng(1)
    x_true = rng.standard_normal(n)
    b = a @ x_true
    x = solve_system(a, b)
    residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert residual < 1e-13


def test_never_forms_the_inverse(monkeypatch):
    # The most direct statement of the rule: if inv is called, the test fails.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.linalg.inv was called")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    a = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(solve_system(a, np.array([2.0, 4.0])), [1.0, 1.0])


def test_singular_matrix_raises():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    try:
        solve_system(a, np.array([1.0, 2.0]))
    except np.linalg.LinAlgError:
        return
    raise AssertionError("a singular system must raise LinAlgError, not answer")


def test_condition_report():
    cond, ok = condition_report(np.eye(3))
    assert cond == 1.0 and ok
    hilbert = 1.0 / (np.arange(14)[:, None] + np.arange(14)[None, :] + 1.0)
    cond, ok = condition_report(hilbert)
    assert cond > 1e12 and not ok
