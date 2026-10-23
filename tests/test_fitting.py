"""fit_least_squares against cases with known answers: straight line, Gaussian, complex delay, scaling, bounds, priors."""

import numpy as np
import pytest

from cjcfit import FitParameter, add_noise, fit_least_squares, fit_quality

X_LINE = np.linspace(0.0, 10.0, 11)
SLOPE, INTERCEPT = 2.0, 1.0
LINE_PARAMETERS = [FitParameter("slope", 1.0), FitParameter("intercept", 0.0)]


def line(x, slope, intercept):
    """Straight line."""
    return slope * x + intercept


def gaussian(x, amplitude, center, width):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def delayed(freq_hz, amplitude, delay):
    """An attenuated, delayed copy: A exp(-i 2 pi f tau)."""
    return amplitude * np.exp(-2j * np.pi * freq_hz * delay)


def decay(time, rate):
    """Exponential decay exp(-rate t)."""
    return np.exp(-rate * time)


def textbook_line_uncertainties(x, sigma):
    """sigma_slope and sigma_intercept of a straight-line fit with known sigma (e.g. Taylor, eq. 8.16-8.17)."""
    n, sum_x, sum_xx = x.size, x.sum(), (x**2).sum()
    delta = n * sum_xx - sum_x**2
    return sigma * np.sqrt(n / delta), sigma * np.sqrt(sum_xx / delta)


def test_line_without_noise_is_recovered_exactly():
    """Noise-free line data give the slope and intercept to rounding, chi-squared 0."""
    fit = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), LINE_PARAMETERS)
    assert fit.values["slope"] == pytest.approx(SLOPE, abs=1e-10)
    assert fit.values["intercept"] == pytest.approx(INTERCEPT, abs=1e-10)
    assert fit.success and fit.chi_squared < 1e-18 and fit.parameter_names == ("slope", "intercept")


def test_line_uncertainties_equal_the_textbook_formulas():
    """With known sigma the uncertainties and correlation are the textbook ones, whatever the noise drawn."""
    sigma = 0.5
    data = add_noise(line(X_LINE, SLOPE, INTERCEPT), additive=sigma, seed=1)
    fit = fit_least_squares(line, X_LINE, data, LINE_PARAMETERS, sigma=sigma)
    sigma_slope, sigma_intercept = textbook_line_uncertainties(X_LINE, sigma)
    assert fit.uncertainties["slope"] == pytest.approx(sigma_slope, rel=1e-6)
    assert fit.uncertainties["intercept"] == pytest.approx(sigma_intercept, rel=1e-6)
    # Correlation of slope and intercept: -sum(x) / sqrt(N sum(x^2)).
    assert fit.correlation[0, 1] == pytest.approx(-X_LINE.sum() / np.sqrt(X_LINE.size * (X_LINE**2).sum()), rel=1e-6)
    # Values within 3 sigma of the truth, and the closed-form least-squares solution exactly.
    assert abs(fit.values["slope"] - SLOPE) < 3 * sigma_slope
    assert np.allclose([fit.values["slope"], fit.values["intercept"]], np.polyfit(X_LINE, data, 1), rtol=1e-8)
    assert fit.degrees_of_freedom == 9


def test_line_without_sigma_estimates_it_from_the_residuals():
    """Without sigma the uncertainties are the textbook ones with sigma^2 = sum(r^2) / (N - 2)."""
    data = add_noise(line(X_LINE, SLOPE, INTERCEPT), additive=0.5, seed=2)
    fit = fit_least_squares(line, X_LINE, data, LINE_PARAMETERS)
    sigma_estimate = np.sqrt(np.sum(fit.residuals**2) / (X_LINE.size - 2))
    sigma_slope, sigma_intercept = textbook_line_uncertainties(X_LINE, sigma_estimate)
    assert fit.uncertainties["slope"] == pytest.approx(sigma_slope, rel=1e-6)
    assert fit.uncertainties["intercept"] == pytest.approx(sigma_intercept, rel=1e-6)


def test_gaussian_peak_recovered_within_uncertainty():
    """Position, width and amplitude of a noisy Gaussian lie within 3 sigma of the truth; reduced chi^2 near 1."""
    x = np.linspace(-5.0, 5.0, 201)
    data = add_noise(gaussian(x, 1.0, 0.3, 0.8), additive=0.02, seed=3)
    parameters = [FitParameter("amplitude", 0.7), FitParameter("center", 0.0), FitParameter("width", 1.2, bounds=(0.0, None))]
    fit = fit_least_squares(gaussian, x, data, parameters, sigma=0.02)
    for name, true in [("amplitude", 1.0), ("center", 0.3), ("width", 0.8)]:
        assert abs(fit.values[name] - true) < 3 * fit.uncertainties[name]
    # 198 degrees of freedom: reduced chi^2 has a spread of sqrt(2/198) = 0.1.
    assert fit.reduced_chi_squared == pytest.approx(1.0, abs=0.3)


def test_complex_amplitude_and_delay():
    """A exp(-i 2 pi f tau) with complex noise: A and tau within 3 sigma, and their analytic uncertainties."""
    freq_hz = np.linspace(0.1e12, 2.0e12, 200)
    sigma = 0.01
    data = add_noise(delayed(freq_hz, 0.6, 2.3e-12), additive=sigma, seed=4)
    parameters = [FitParameter("amplitude", 0.5), FitParameter("delay", 2.25e-12, scale="normalized", normalization_factor=1e-12)]
    fit = fit_least_squares(delayed, freq_hz, data, parameters, sigma=sigma)
    assert abs(fit.values["amplitude"] - 0.6) < 3 * fit.uncertainties["amplitude"]
    assert abs(fit.values["delay"] - 2.3e-12) < 3 * fit.uncertainties["delay"]
    # Each complex point is two measurements with sigma: sigma_A = sigma / sqrt(N),
    # sigma_tau = sigma / (A 2 pi sqrt(sum f^2)), and A and tau are uncorrelated.
    assert fit.uncertainties["amplitude"] == pytest.approx(sigma / np.sqrt(freq_hz.size), rel=1e-3)
    expected_delay = sigma / (fit.values["amplitude"] * 2 * np.pi * np.sqrt(np.sum(freq_hz**2)))
    assert fit.uncertainties["delay"] == pytest.approx(expected_delay, rel=1e-3)
    assert abs(fit.correlation[0, 1]) < 1e-3
    assert np.iscomplexobj(fit.model_curve) and fit.degrees_of_freedom == 2 * 200 - 2


def test_complex_sigma_weights_real_and_imaginary_parts_separately():
    """Complex sigma divides Re and Im residuals by its own part each."""
    freq_hz = np.linspace(0.1e12, 2.0e12, 50)
    data = add_noise(delayed(freq_hz, 0.6, 2.3e-12), additive=0.01 + 0.03j, seed=5)
    parameters = [FitParameter("amplitude", 0.6), FitParameter("delay", 2.3e-12, scale="normalized", normalization_factor=1e-12)]
    fit = fit_least_squares(delayed, freq_hz, data, parameters, sigma=0.01 + 0.03j)
    expected = fit.residuals.real / 0.01 + 1j * fit.residuals.imag / 0.03
    assert np.allclose(fit.normalized_residuals, expected)


def test_log10_scale_reaches_a_parameter_decades_away():
    """A decay rate started 4 decades too low is found with scale='log10'."""
    time = np.logspace(-9, -3, 60)
    parameters = [FitParameter("rate", 1e2, scale="log10", bounds=(1.0, 1e12))]
    fit = fit_least_squares(decay, time, decay(time, 1e6), parameters)
    assert fit.values["rate"] == pytest.approx(1e6, rel=1e-6)


def test_uncertainty_in_physical_units_does_not_depend_on_the_scale():
    """The same noisy decay fitted with and without log10 scaling reports the same physical uncertainty."""
    time = np.logspace(-8, -5, 40)
    data = add_noise(decay(time, 1e6), additive=0.01, seed=6)
    log_fit = fit_least_squares(decay, time, data, [FitParameter("rate", 1e5, scale="log10")], sigma=0.01)
    linear_fit = fit_least_squares(decay, time, data, [FitParameter("rate", log_fit.values["rate"])], sigma=0.01)
    assert log_fit.values["rate"] == pytest.approx(linear_fit.values["rate"], rel=1e-6)
    assert log_fit.uncertainties["rate"] == pytest.approx(linear_fit.uncertainties["rate"], rel=1e-4)


def test_bounds_are_respected():
    """A slope that may not exceed 1.5 stops there and is flagged as at its bound."""
    parameters = [FitParameter("slope", 1.0, bounds=(None, 1.5)), FitParameter("intercept", 0.0)]
    fit = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), parameters)
    assert fit.values["slope"] == pytest.approx(1.5)
    # at_bound is one OR over all parameters and both bounds; parameters_at_bound says which.
    assert fit.at_bound is True and fit.parameters_at_bound == ("slope",)
    free = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), LINE_PARAMETERS)
    assert free.at_bound is False and free.parameters_at_bound == ()


def test_lower_bound_counts_as_well():
    """A parameter stopped by its lower bound also makes at_bound True."""
    parameters = [FitParameter("slope", 3.0, bounds=(2.5, None)), FitParameter("intercept", 0.0)]
    fit = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), parameters)
    assert fit.at_bound is True and fit.parameters_at_bound == ("slope",)


def test_r_squared_is_reported_and_matches_fit_quality():
    """R^2 is 1 for a perfect fit and equals fit_quality's R^2 of the same model curve, real and complex."""
    exact = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), LINE_PARAMETERS)
    assert exact.r_squared == pytest.approx(1.0)
    data = add_noise(line(X_LINE, SLOPE, INTERCEPT), additive=0.5, seed=8)
    noisy = fit_least_squares(line, X_LINE, data, LINE_PARAMETERS, sigma=0.5)
    assert noisy.r_squared == pytest.approx(fit_quality(data, noisy.model_curve).r_squared)
    assert 0.9 < noisy.r_squared < 1.0
    freq_hz = np.linspace(0.1e12, 2.0e12, 50)
    complex_data = add_noise(delayed(freq_hz, 0.6, 2.3e-12), additive=0.05, seed=9)
    parameters = [FitParameter("amplitude", 0.6), FitParameter("delay", 2.3e-12, scale="normalized", normalization_factor=1e-12)]
    complex_fit = fit_least_squares(delayed, freq_hz, complex_data, parameters, sigma=0.05)
    # For complex data R^2 = 1 - sum|r|^2 / sum|d - mean(d)|^2.
    expected = 1 - np.sum(np.abs(complex_fit.residuals) ** 2) / np.sum(np.abs(complex_data - complex_data.mean()) ** 2)
    assert complex_fit.r_squared == pytest.approx(expected)


def test_prior_combines_with_the_data_as_inverse_variance_weights():
    """A constant level with a prior: the textbook weighted mean of data and prior, and the combined sigma."""
    sigma, n_points = 0.4, 16
    data = add_noise(np.full(n_points, 3.0), additive=sigma, seed=7)
    parameters = [FitParameter("level", 1.0, prior_mean=2.5, prior_std=0.2)]
    fit = fit_least_squares(lambda x, level: np.full(n_points, level), None, data, parameters, sigma=sigma)
    # Data alone: mean +- sigma / sqrt(N) = +-0.1; prior: 2.5 +- 0.2. Weights 1/0.1^2 = 100 and 1/0.2^2 = 25.
    expected = (100 * data.mean() + 25 * 2.5) / 125
    assert fit.values["level"] == pytest.approx(expected, rel=1e-8)
    assert fit.uncertainties["level"] == pytest.approx(1 / np.sqrt(125), rel=1e-6)
    # cost = chi^2 + the prior term.
    assert fit.cost == pytest.approx(fit.chi_squared + ((fit.values["level"] - 2.5) / 0.2) ** 2)


def test_prior_fixes_a_parameter_the_data_do_not_determine():
    """Without a prior an unused parameter has an infinite uncertainty (with a warning); a prior sets value and spread."""
    def model(x, slope, offset):  # offset has no effect on the curve
        return slope * x

    data = line(X_LINE, SLOPE, 0.0)
    with pytest.warns(RuntimeWarning, match="covariance could not be estimated"):
        free = fit_least_squares(model, X_LINE, data, [FitParameter("slope", 1.0), FitParameter("offset", 0.0)], sigma=0.1)
    assert np.isinf(free.uncertainties["offset"])
    parameters = [FitParameter("slope", 1.0), FitParameter("offset", 0.0, prior_mean=0.7, prior_std=0.05)]
    fit = fit_least_squares(model, X_LINE, data, parameters, sigma=0.1)
    assert fit.values["offset"] == pytest.approx(0.7) and fit.uncertainties["offset"] == pytest.approx(0.05)


def test_diff_step_lets_a_fit_see_through_numerical_noise():
    """A model rounded to 1e-6 has a zero Jacobian at the default step; diff_step=1e-3 fits it."""
    def rounded_line(x, slope):
        return np.round(slope * x / 1e-6) * 1e-6

    data = rounded_line(X_LINE, SLOPE)
    fit = fit_least_squares(rounded_line, X_LINE, data, [FitParameter("slope", 1.0, diff_step=1e-3)])
    assert fit.values["slope"] == pytest.approx(SLOPE, abs=1e-5)


def test_initial_values_replace_the_start():
    """initial_values starts the fit elsewhere and refuses names that are not fitted."""
    fit = fit_least_squares(line, X_LINE, line(X_LINE, SLOPE, INTERCEPT), LINE_PARAMETERS, initial_values={"slope": 2.0})
    assert fit.values["slope"] == pytest.approx(SLOPE)
    with pytest.raises(ValueError, match="not fitted"):
        fit_least_squares(line, X_LINE, X_LINE, LINE_PARAMETERS, initial_values={"slop": 2.0})


@pytest.mark.parametrize(
    ("model", "data", "kwargs", "message"),
    [
        (lambda x, slope, intercept: slope * x[:5], X_LINE, {}, "shape"),
        (lambda x, slope, intercept: slope * x / 0.0, X_LINE, {}, "NaN or inf at the start"),
        (lambda x, slope, intercept: slope * x + 1j, X_LINE, {}, "non-zero imaginary part"),
        (line, X_LINE, {"sigma": 0.0}, "sigma must be positive"),
        (line, X_LINE, {"sigma": 0.1j}, "sigma must be real"),
        (line, X_LINE, {"least_squares_options": {"bounds": (0, 1)}}, "set from the FitParameters"),
    ],
)
def test_invalid_input_raises_clear_errors(model, data, kwargs, message):
    """Wrong model output, a model that fails at the start, a bad sigma or reserved options raise a ValueError."""
    with np.errstate(divide="ignore", invalid="ignore"), pytest.raises(ValueError, match=message):
        fit_least_squares(model, X_LINE, data, LINE_PARAMETERS, **kwargs)


def test_duplicate_names_are_refused():
    """Two parameters with the same name raise a ValueError."""
    with pytest.raises(ValueError, match="unique"):
        fit_least_squares(line, X_LINE, X_LINE, [FitParameter("slope", 1.0), FitParameter("slope", 2.0)])
