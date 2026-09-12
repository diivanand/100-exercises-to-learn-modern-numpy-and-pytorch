# =============================================================================
#  04.03 -- Least squares
# =============================================================================
#
#  A square system has (with luck) one solution. A tall one, more equations
#  than unknowns, generally has none: the data is noisy and no line goes
#  through every point. Least squares asks instead for the x that makes
#  A x - b as short as possible, and `np.linalg.lstsq(A, b, rcond=None)`
#  computes it (by SVD, so it is happy with rank-deficient A too). Johansson,
#  *Numerical Python* ch. 5, "Rectangular Systems", covers over- and
#  under-determined systems, and ch. 14 uses lstsq as the engine under every
#  linear regression.
#
#  The part people get wrong is not the solver but the DESIGN MATRIX. Each
#  column is one unknown's contribution. For a straight line y = m x + c the
#  unknowns are m and c, so the matrix needs a column of x values AND a
#  column of ones. Forget the ones and you have asked for the best line
#  through the origin, which the solver will cheerfully provide.
#
#  For polynomials there is a better tool than building the Vandermonde
#  matrix yourself: `numpy.polynomial.Polynomial.fit(x, y, degree)`. It scales
#  x into [-1, 1] before solving, which is why its results stay accurate
#  where a raw Vandermonde fit falls apart (04.02 again). Johansson ch. 7
#  notes that `numpy.polynomial` supersedes the old np.polyfit / np.poly1d
#  pair, whose coefficients are in the opposite order; 05.06 is about that.
#
#  On `rcond=None`: the parameter controls which tiny singular values count
#  as zero. The old default was machine epsilon; `None` means "epsilon times
#  the larger dimension", which is what you want, and has been the default
#  since NumPy 2.0. Passing it explicitly documents that you know.
#
#  TASK
#    Give `fit_line` a proper design matrix, and implement `fit_polynomial`
#    with Polynomial.fit.
#
#  RUN IT
#    ./npt test 04_03
#
# =============================================================================

import numpy as np
from numpy.polynomial import Polynomial


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of y ~ slope * x + intercept."""
    # TODO: the design matrix has only the x column, so the intercept is
    # never fitted -- it is silently pinned to zero. Add the column of ones
    # (np.column_stack, or np.vander(x, 2)).
    design = x[:, None]
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design, y, rcond=None
    )
    return float(coefficients[0]), 0.0


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> Polynomial:
    """Least-squares polynomial of the given degree, as a Polynomial object."""
    # TODO: use Polynomial.fit. This returns the wrong type on purpose so
    # that the last test tells you what the interface is.
    return np.polyfit(x, y, degree)


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
