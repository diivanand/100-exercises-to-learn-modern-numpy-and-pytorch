# =============================================================================
#  05.06 -- Polynomials
# =============================================================================
#
#  NumPy has two polynomial APIs, and they disagree about the one thing that
#  matters: coefficient order.
#
#      np.polyfit / np.poly1d / np.polyval    HIGHEST degree first: [c2, c1, c0]
#      numpy.polynomial.Polynomial            LOWEST degree first:  [c0, c1, c2]
#
#  Johansson, *Numerical Python* ch. 7, "Polynomials", puts it plainly: the
#  two "are incompatible", "the coordinate arrays have reversed order", and
#  `numpy.poly1d` "has been superseded by numpy.polynomial, which is now
#  recommended for new code". The old functions still exist, still appear
#  in answers online, and are the source of a whole genre of off-by-reverse
#  bugs. Use the class.
#
#  `Polynomial([c0, c1, c2])` reads like the formula. `Polynomial.fit(x, y,
#  degree)` is least squares (04.03) in a scaled domain, which is why it
#  stays accurate for x values in the thousands where np.polyfit warns about
#  conditioning; call `.convert()` to get coefficients in plain powers of x.
#  `.roots()`, `.deriv()`, `.integ()`, `p(x)`, and arithmetic between
#  polynomials all just work.
#
#  TASK
#    Build `quadratic` as a Polynomial and produce the report with the
#    standard-basis coefficients in the documented order.
#
#  RUN IT
#    ./npt test 05_06
#
# =============================================================================

import numpy as np
from numpy.polynomial import Polynomial


def quadratic(c0: float, c1: float, c2: float) -> Polynomial:
    """c0 + c1 x + c2 x^2 as a Polynomial object."""
    # TODO: this is the poly1d order, and the wrong class.
    return np.poly1d([c0, c1, c2])


def fit_and_report(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, np.ndarray]:
    """Fit a polynomial and report its standard-basis coefficients (lowest
    degree first), its real roots, and its derivative's coefficients."""
    # TODO: np.polyfit returns highest-degree-first coefficients and is
    # ill-conditioned for x far from zero. Use Polynomial.fit(...).convert().
    coef = np.polyfit(x, y, degree)
    p = np.poly1d(coef)
    roots = p.roots
    return {
        "coef": coef,
        "real_roots": np.sort(roots[np.isreal(roots)].real),
        "deriv_coef": p.deriv().coeffs,
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
