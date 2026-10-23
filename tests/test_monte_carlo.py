"""monte_carlo_noise_fit and monte_carlo_convergence_fit: statistics, reproducibility, process count, failures.

The models are defined at module level, because the runs with n_processes > 1 send them to other processes.
"""

import numpy as np
import pytest

from cjcfit import FitParameter, MonteCarloResult, monte_carlo_convergence_fit, monte_carlo_noise_fit

X = np.linspace(0.0, 10.0, 11)
LINE_PARAMETERS = [FitParameter("slope", 1.0), FitParameter("intercept", 0.0)]
TRUE_LINE = {"slope": 2.0, "intercept": 1.0}


def line(x, slope, intercept):
    """Straight line."""
    return slope * x + intercept


def squared_slope(x, root):
    """root^2 * x: the same curve for +root and -root, so chi^2 has two equal minima."""
    return root**2 * x


def refusing_line(x, slope, intercept):
    """A line that raises an error away from the true values (to test failed runs)."""
    if slope != 2.0:
        raise RuntimeError("model broken here")
    return line(x, slope, intercept)


def test_noise_fit_spread_matches_reported_uncertainty_and_bias_is_zero():
    """Linear model with sigma 0.5: spread = textbook sigma_slope = reported uncertainty, bias within its error."""
    n_runs, sigma = 400, 0.5
    result = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, additive=sigma, sigma=sigma, n_runs=n_runs, seed=11)
    sum_x, sum_xx = X.sum(), (X**2).sum()
    sigma_slope = sigma * np.sqrt(X.size / (X.size * sum_xx - sum_x**2))
    # With sigma given every fit reports exactly the textbook uncertainty.
    assert result.median_uncertainty["slope"] == pytest.approx(sigma_slope, rel=1e-6)
    # The sample std of 400 runs scatters by 1/sqrt(2 * 399) = 3.5 %: allow 4 of those.
    assert result.std["slope"] == pytest.approx(sigma_slope, rel=0.14)
    # The bias of an unbiased fit scatters by std / sqrt(n_runs): allow 4 of those.
    for name in TRUE_LINE:
        assert abs(result.bias[name]) < 4 * result.std[name] / np.sqrt(n_runs)
    assert result.n_failed == 0 and result.values["slope"].shape == (n_runs,)
    assert result.n_at_bound == 0 and result.n_at_bound_per_parameter == {"slope": 0, "intercept": 0}
    assert result.true_values == TRUE_LINE and not np.any(result.at_bound)
    # Every run reports its R^2; with noise 0.5 on a line rising by 20 it stays close to 1.
    assert result.r_squared.shape == (n_runs,) and np.all(result.r_squared > 0.95)


def test_noise_fit_is_reproducible_and_independent_of_the_process_count():
    """The same seed gives identical runs with 1 and with 2 processes; the stored seed repeats a study without one."""
    settings = {"additive": 0.5, "n_runs": 6}
    one = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, **settings, seed=5, n_processes=1)
    two = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, **settings, seed=5, n_processes=2)
    for name in TRUE_LINE:
        assert np.array_equal(one.values[name], two.values[name])
        assert np.array_equal(one.uncertainties[name], two.uncertainties[name])
    unseeded = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, **settings)
    again = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, **settings, seed=unseeded.seed)
    assert np.array_equal(unseeded.values["slope"], again.values["slope"])


def test_runs_that_raise_count_as_failed():
    """A model error inside the runs does not stop the study: the runs are failed, NaN, with the message kept."""
    result = monte_carlo_noise_fit(refusing_line, X, TRUE_LINE, LINE_PARAMETERS, additive=0.1, n_runs=3, seed=1)
    assert result.n_failed == 3 and np.all(np.isnan(result.values["slope"])) and np.isnan(result.mean["slope"])
    assert "model broken here" in result.first_error


def test_noise_fit_checks_the_true_values():
    """true_values must name exactly the fitted parameters."""
    with pytest.raises(ValueError, match="exactly the fitted parameters"):
        monte_carlo_noise_fit(line, X, {"slope": 2.0}, LINE_PARAMETERS, additive=0.1, n_runs=2)


def test_convergence_fit_finds_both_minima_from_wide_starts_and_one_from_narrow_starts():
    """root^2 x fitted to 4 x: wide starts end at both -2 and +2, narrow starts around 1.5 only at +2."""
    data = squared_slope(X, 2.0)
    wide = monte_carlo_convergence_fit(
        squared_slope, X, data, [FitParameter("root", 0.5, initial_std=3.0)], sigma=0.1, n_runs=40, seed=3
    )
    ends = np.round(wide.values["root"], 6)
    assert set(ends) == {-2.0, 2.0}
    # Both minima are equally deep: every run ends with (essentially) zero cost.
    assert np.all(wide.cost < 1e-12) and wide.n_failed == 0
    narrow = monte_carlo_convergence_fit(
        squared_slope, X, data, [FitParameter("root", 1.5, initial_std=0.2)], sigma=0.1, n_runs=20, seed=3
    )
    assert np.allclose(narrow.values["root"], 2.0) and narrow.best_values["root"] == pytest.approx(2.0)


def test_convergence_starts_follow_scale_and_bounds():
    """Log-scale starts are log-normal with initial_std / (initial ln 10) decades; bounded starts stay inside."""
    data = line(X, 2.0, 1.0)
    parameters = [
        FitParameter("slope", 2.0, scale="log10", initial_std=0.2),
        FitParameter("intercept", 1.0, bounds=(0.9, 1.1), initial_std=1.0),
    ]
    result = monte_carlo_convergence_fit(line, X, data, parameters, n_runs=400, seed=8)
    log_starts = np.log10(result.start_values["slope"])
    # 0.2 / (2 ln 10) = 0.0434 decades; the sample std of 400 draws scatters by 3.5 %.
    assert np.std(log_starts) == pytest.approx(0.2 / (2.0 * np.log(10.0)), rel=0.14)
    assert np.all((result.start_values["intercept"] >= 0.9) & (result.start_values["intercept"] <= 1.1))
    # Every run reaches the one minimum.
    assert np.allclose(result.values["slope"], 2.0) and np.allclose(result.values["intercept"], 1.0)


def test_invalid_counts_raise():
    """n_runs and n_processes must be positive integers."""
    with pytest.raises(ValueError, match="n_runs"):
        monte_carlo_convergence_fit(line, X, X, LINE_PARAMETERS, n_runs=0)
    with pytest.raises(ValueError, match="n_processes"):
        monte_carlo_convergence_fit(line, X, X, LINE_PARAMETERS, n_runs=2, n_processes=0)


def test_noise_fit_with_random_start_values():
    """random_start draws starts from initial +- initial_std; the noise per run stays the same, and a line ends in the same place."""
    parameters = [FitParameter("slope", 1.0, initial_std=0.5), FitParameter("intercept", 0.0, initial_std=2.0)]
    settings = {"additive": 0.5, "sigma": 0.5, "n_runs": 300, "seed": 4}
    fixed = monte_carlo_noise_fit(line, X, TRUE_LINE, parameters, **settings)
    random = monte_carlo_noise_fit(line, X, TRUE_LINE, parameters, **settings, random_start=True)
    # Without random_start every run starts at initial.
    assert np.all(fixed.start_values["slope"] == 1.0) and np.all(fixed.start_values["intercept"] == 0.0)
    # With it the starts spread by initial_std (the sample std of 300 draws scatters by 4 %: allow 4 of those).
    assert np.std(random.start_values["slope"]) == pytest.approx(0.5, rel=0.16)
    assert np.std(random.start_values["intercept"]) == pytest.approx(2.0, rel=0.16)
    # A line has one minimum, and the noise of each run does not depend on the starts: the same end points.
    assert np.allclose(random.values["slope"], fixed.values["slope"], rtol=1e-6)


def test_random_start_shows_a_second_minimum_in_the_noise_study():
    """root^2 x with starts around 0: some runs end at -2, which a fixed start at +1.5 never shows."""
    parameters = [FitParameter("root", 1.5, initial_std=2.0)]
    settings = {"additive": 0.1, "sigma": 0.1, "n_runs": 60, "seed": 6}
    fixed = monte_carlo_noise_fit(squared_slope, X, {"root": 2.0}, parameters, **settings)
    random = monte_carlo_noise_fit(squared_slope, X, {"root": 2.0}, parameters, **settings, random_start=True)
    assert np.all(fixed.values["root"] > 0)
    assert np.any(random.values["root"] < 0) and abs(random.bias["root"]) > 10 * abs(fixed.bias["root"])


def test_random_start_is_independent_of_the_process_count():
    """Random starts and their results are identical with 1 and with 2 processes."""
    parameters = [FitParameter("slope", 1.0, initial_std=0.5), FitParameter("intercept", 0.0, initial_std=2.0)]
    settings = {"additive": 0.5, "n_runs": 6, "seed": 5, "random_start": True}
    one = monte_carlo_noise_fit(line, X, TRUE_LINE, parameters, **settings, n_processes=1)
    two = monte_carlo_noise_fit(line, X, TRUE_LINE, parameters, **settings, n_processes=2)
    assert np.array_equal(one.start_values["slope"], two.start_values["slope"])
    assert np.array_equal(one.values["slope"], two.values["slope"])


def test_at_bound_per_run_is_one_or_like_fit_result():
    """A bound at the true slope: at_bound is True per run when any parameter sits on any bound, with its name."""
    parameters = [FitParameter("slope", 1.0, bounds=(None, 1.9)), FitParameter("intercept", 0.0)]
    result = monte_carlo_noise_fit(line, X, TRUE_LINE, parameters, additive=0.1, n_runs=20, seed=2)
    # The true slope 2.0 lies beyond the bound 1.9, so every run stops there.
    assert result.at_bound.dtype == bool and np.all(result.at_bound)
    assert all(names == ("slope",) for names in result.parameters_at_bound)
    assert result.n_at_bound == 20 and result.n_at_bound_per_parameter == {"slope": 20, "intercept": 0}


def test_both_studies_return_the_same_result_class():
    """Noise and convergence studies give a MonteCarloResult; a convergence study has no truth and no bias."""
    noise = monte_carlo_noise_fit(line, X, TRUE_LINE, LINE_PARAMETERS, additive=0.1, n_runs=3, seed=1)
    convergence = monte_carlo_convergence_fit(line, X, line(X, 2.0, 1.0), LINE_PARAMETERS, n_runs=3, seed=1)
    assert type(noise) is type(convergence) is MonteCarloResult
    assert convergence.true_values is None and convergence.bias is None
    assert noise.best_values["slope"] == pytest.approx(2.0, abs=0.1) and convergence.best_cost == pytest.approx(0.0, abs=1e-12)
