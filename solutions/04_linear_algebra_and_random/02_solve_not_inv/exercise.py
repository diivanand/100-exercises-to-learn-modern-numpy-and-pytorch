# Solution -- 04.02 Solve, do not invert

import numpy as np


def solve_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a @ x = b. `b` may be a vector (n,) or a stack of columns (n, k)."""
    # One LU factorisation, then forward/back substitution for every column of
    # b. Forming inv(a) costs about three times as much and, for an
    # ill-conditioned a, throws away digits that solve() keeps.
    return np.linalg.solve(a, b)


def condition_report(a: np.ndarray) -> tuple[float, bool]:
    """The 2-norm condition number of `a`, and whether it is safe to solve with.

    "Safe" here means cond(a) < 1e12: with float64's ~16 significant digits,
    that leaves about four digits of accuracy in the solution.
    """
    cond = float(np.linalg.cond(a))
    return cond, bool(np.isfinite(cond) and cond < 1e12)


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
