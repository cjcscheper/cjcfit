"""weights: relative 1/sigma on their own, an extra emphasis with sigma (sandwich covariance), 0 leaves a point out."""

import numpy as np
import pytest

from cjcfit import FitParameter, add_noise, discrete_grid_fit, fit_least_squares, fit_quality, monte_carlo_convergence_fit

X = np.linspace(0.0, 10.0, 21)
LINE_PARAMETERS = [FitParameter("slope", 1.0), FitParameter("intercept", 0.0)]
# Noise that grows along x, so weighting matters.
SIGMA = 0.1 + 0.1 * X


def line(x, slope, intercept):
    """Straight line."""
    return slope * x + intercept


def with_outlier():
    """Noise-free line data with one point 50 too high."""
    data = line(X, 2.0, 1.0)
    data[7] += 50.0
    return data


def test_weights_alone_are_relative_inverse_sigma():
    """weights = 1/sigma gives the values of sigma = sigma, with uncertainties scaled by sqrt(reduced chi^2)."""
    data = add_noise(line(X, 2.0, 1.0), additive=SIGMA, seed=1)
    with_sigma = fit_least_squares(line, X, data, LINE_PARAMETERS, sigma=SIGMA)
    with_weights = fit_least_squares(line, X, data, LINE_PARAMETERS, weights=1.0 / SIGMA)
    scale = np.sqrt(with_sigma.reduced_chi_squared)
    for name in with_sigma.parameter_names:
        assert with_weights.values[name] == pytest.approx(with_sigma.values[name], rel=1e-8)
        assert with_weights.uncertainties[name] == pytest.approx(with_sigma.uncertainties[name] * scale, rel=1e-6)
    # Only the ratios of the weights matter: ten times larger weights change nothing.
    tenfold = fit_least_squares(line, X, data, LINE_PARAMETERS, weights=10.0 / SIGMA)
    assert tenfold.uncertainties["slope"] == pytest.approx(with_weights.uncertainties["slope"], rel=1e-6)
    assert np.allclose(with_weights.normalized_residuals, with_weights.residuals / SIGMA)


def test_weight_zero_leaves_a_point_out():
    """An outlier with weight 0 does not pull the fit, and is not counted in chi^2 or the degrees of freedom."""
    weights = np.ones(X.size)
    weights[7] = 0.0
    fit = fit_least_squares(line, X, with_outlier(), LINE_PARAMETERS, sigma=0.1, weights=weights)
    assert fit.values == pytest.approx({"slope": 2.0, "intercept": 1.0})
    assert fit.chi_squared == pytest.approx(0.0, abs=1e-12) and fit.degrees_of_freedom == X.size - 1 - 2
    # R^2 leaves the outlier out as well: the line explains the remaining points perfectly.
    assert fit.r_squared == pytest.approx(1.0)


def test_sigma_and_weights_give_the_sandwich_covariance():
    """With sigma and weights the covariance is (A^T W A)^-1 A^T W^2 A (A^T W A)^-1 sigma^2, not (A^T W A)^-1."""
    sigma = 0.3
    weights = np.linspace(0.2, 1.0, X.size)  # emphasis on the right end
    data = add_noise(line(X, 2.0, 1.0), additive=sigma, seed=2)
    fit = fit_least_squares(line, X, data, LINE_PARAMETERS, sigma=sigma, weights=weights)
    # Weighted least squares by hand: design matrix A, W = diag(weights^2).
    design = np.column_stack([X, np.ones_like(X)])
    w2 = weights**2
    bread = np.linalg.inv(design.T @ (w2[:, None] * design))
    expected = bread @ design.T @ ((w2**2)[:, None] * design) @ bread * sigma**2
    assert np.allclose(fit.covariance, expected, rtol=1e-6)
    assert np.allclose([fit.values["slope"], fit.values["intercept"]], bread @ design.T @ (w2 * data), rtol=1e-6)
    # chi^2 tests sigma, so it does not contain the weights.
    assert fit.chi_squared == pytest.approx(np.sum((fit.residuals / sigma) ** 2))


def test_complex_weights_weight_real_and_imaginary_parts_separately():
    """Weights 1 + 0j fit Re only: a meaningless imaginary part does not matter."""
    data = line(X, 2.0, 1.0) + 1j * np.random.default_rng(3).normal(0.0, 100.0, X.size)
    fit = fit_least_squares(lambda x, slope, intercept: line(x, slope, intercept) + 0j, X, data, LINE_PARAMETERS, weights=1.0 + 0j)
    assert fit.values == pytest.approx({"slope": 2.0, "intercept": 1.0})
    assert fit.degrees_of_freedom == X.size - 2


@pytest.mark.parametrize(
    ("weights", "message"),
    [(-1.0, "zero or positive"), (0.0, "At least one weight"), (0.5j, "weights must be real"), (np.ones(3), "does not match")],
)
def test_invalid_weights_raise(weights, message):
    """Negative, all-zero, imaginary (on real data) or mis-shaped weights raise a ValueError."""
    with pytest.raises(ValueError, match=message):
        fit_least_squares(line, X, X, LINE_PARAMETERS, weights=weights)


def test_grid_quality_and_monte_carlo_take_the_same_weights():
    """discrete_grid_fit, fit_quality and monte_carlo_convergence_fit leave a weight-0 outlier out too."""
    weights = np.ones(X.size)
    weights[7] = 0.0
    data = with_outlier()
    grid = discrete_grid_fit(line, X, data, {"slope": [1.5, 2.0, 2.5], "intercept": [0.0, 1.0, 2.0]}, weights=weights)
    assert grid.best_values == {"slope": 2.0, "intercept": 1.0} and grid.best_chi_squared == pytest.approx(0.0)
    quality = fit_quality(data, line(X, 2.0, 1.0), sigma=0.1, weights=weights, n_parameters=2)
    assert quality.chi_squared == pytest.approx(0.0) and quality.degrees_of_freedom == X.size - 1 - 2
    runs = monte_carlo_convergence_fit(line, X, data, LINE_PARAMETERS, weights=weights, n_runs=3, seed=1)
    assert np.allclose(runs.values["slope"], 2.0)
