# Solution -- 05.06 Polynomials

import numpy as np
from numpy.polynomial import Polynomial


def quadratic(c0: float, c1: float, c2: float) -> Polynomial:
    """c0 + c1 x + c2 x^2 as a Polynomial object."""
    # numpy.polynomial stores coefficients LOWEST degree first, so the
    # constructor argument reads like the formula.
    return Polynomial([c0, c1, c2])


def fit_and_report(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, np.ndarray]:
    """Fit a polynomial and report its standard-basis coefficients (lowest
    degree first), its real roots, and its derivative's coefficients."""
    # Polynomial.fit works in a scaled domain for conditioning; convert()
    # maps the result back to plain powers of x so the coefficients mean
    # what the caller expects.
    p = Polynomial.fit(x, y, degree).convert()
    roots = p.roots()
    return {
        "coef": p.coef,
        "real_roots": np.sort(roots[np.isreal(roots)].real),
        "deriv_coef": p.deriv().coef,
    }


def test_quadratic_evaluates_correctly():
    p = quadratic(2.0, -3.0, 1.0)  # (x - 1)(x - 2) = 2 - 3x + x^2
    np.testing.assert_allclose(p(np.array([0.0, 1.0, 2.0, 3.0])), [2.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(np.sort(p.roots()), [1.0, 2.0])


def test_coefficients_are_lowest_degree_first():
    p = quadratic(5.0, 0.0, 1.0)  # 5 + x^2
    np.testing.assert_allclose(p.coef, [5.0, 0.0, 1.0])
    assert p(2.0) == 9.0


def test_fit_and_report():
    x = np.linspace(-3.0, 3.0, 61)
    y = (x - 1.0) * (x + 2.0)  # x^2 + x - 2
    report = fit_and_report(x, y, 2)
    np.testing.assert_allclose(report["coef"], [-2.0, 1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(report["real_roots"], [-2.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(report["deriv_coef"], [1.0, 2.0], atol=1e-10)


def test_fit_is_well_conditioned_on_far_away_x():
    # x values around 1e4: a raw Vandermonde fit (np.polyfit) warns about
    # conditioning here; Polynomial.fit's domain scaling does not care.
    x = np.linspace(10_000.0, 10_010.0, 50)
    y = 3.0 + 0.5 * (x - 10_005.0)
    report = fit_and_report(x, y, 1)
    np.testing.assert_allclose(report["deriv_coef"], [0.5], atol=1e-8)
