# Solution -- 04.03 Least squares

import numpy as np
from numpy.polynomial import Polynomial


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of y ~ slope * x + intercept."""
    # The design matrix has one column per unknown: the x values for the
    # slope and a column of ones for the intercept. Leave the ones out and
    # you have fitted a line through the origin, whatever the data says.
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design, y, rcond=None
    )
    slope, intercept = coefficients
    return float(slope), float(intercept)


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> Polynomial:
    """Least-squares polynomial of the given degree, as a Polynomial object."""
    # Polynomial.fit scales x into [-1, 1] before solving, which keeps the
    # design matrix well conditioned (a Vandermonde matrix on raw x values is
    # exactly the ill-conditioned case of 04.02). The returned object carries
    # that scaling in its `domain`, so evaluating it just works.
    return Polynomial.fit(x, y, degree)


def test_fit_line_recovers_slope_and_intercept():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 200)
    y = 3.0 * x - 7.0 + 0.01 * rng.standard_normal(x.size)
    slope, intercept = fit_line(x, y)
    assert abs(slope - 3.0) < 1e-2
    assert abs(intercept + 7.0) < 5e-2


def test_fit_line_is_exact_on_exact_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 5.0
    slope, intercept = fit_line(x, y)
    np.testing.assert_allclose([slope, intercept], [2.0, 5.0], atol=1e-12)


def test_fit_line_does_not_force_the_origin():
    # Data that is flat and far from zero. A fit without an intercept column
    # would report a positive slope here, because a line through the origin
    # is the only thing it can draw.
    x = np.linspace(1.0, 2.0, 50)
    y = np.full_like(x, 100.0)
    slope, intercept = fit_line(x, y)
    assert abs(slope) < 1e-9
    assert abs(intercept - 100.0) < 1e-9


def test_fit_polynomial_matches_the_generating_polynomial():
    x = np.linspace(-2.0, 2.0, 41)
    y = 1.0 - 2.0 * x + 0.5 * x**2
    p = fit_polynomial(x, y, 2)
    np.testing.assert_allclose(p(x), y, atol=1e-10)
    # Coefficients in the standard basis, lowest degree first. `.convert()`
    # undoes the internal domain scaling that Polynomial.fit applies.
    np.testing.assert_allclose(p.convert().coef, [1.0, -2.0, 0.5], atol=1e-10)


def test_fit_polynomial_returns_a_polynomial_object():
    x = np.linspace(0.0, 1.0, 10)
    p = fit_polynomial(x, x**3, 3)
    assert isinstance(p, Polynomial)
    assert p.degree() == 3
