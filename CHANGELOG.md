# Changelog

All notable changes to cjcfit. Versions follow MAJOR.MINOR.PATCH: patch for
fixes, minor for new features, major for changes that break existing scripts.

## [Unreleased]

### Changed
- README: the example figures are grouped per script, each with a short
  explanation (Gaussian peak, complex data, noise models, grid, Monte Carlo).
- `example_noise_models.py`: the legend of each noisy figure names its noise
  form (σ = 0.2, σ: step, σ: bathtub), so each figure can be read on its own.

### Removed
- `example_lab_c_noise.py` and its figures: an instrument calibration does not
  belong in a general fitting package. `example_noise_models.py` still shows
  how to pass your own noise function (a bathtub) to `add_noise`.

## [0.1.0] - 2026-10-23

First release.

### Added
- `FitParameter` (frozen dataclass): start value, bounds (either side open),
  `scale` = None / "log10" / "ln" / "normalized" (with `normalization_factor`),
  Gaussian prior (`prior_mean`, `prior_std`), absolute `diff_step` and
  `initial_std`, all in physical units and checked when created.
  `to_scaled`, `to_physical` and `suggest_diff_step` as plain functions.
- `fit_least_squares(model, x, data, parameters, ...)` for models with named
  arguments, real or complex data, real or complex `sigma` and `weights`
  (weight 0 leaves a point out; weights alone are relative 1/sigma; with
  sigma the covariance is the sandwich formula), start values from
  `initial_values`; returns a frozen `FitResult` with values, uncertainties,
  covariance, correlation, `at_bound` (True if any parameter ended on any
  bound) with `parameters_at_bound`, chi-squared, reduced chi-squared, R², cost, the model curve and the (normalized) residuals. The
  covariance is absolute with `sigma` and scaled by the reduced chi-squared
  without it; a parameter the data do not determine gets an infinite
  uncertainty and a warning.
- `sigma` and `weights` also in `fit_quality`, `discrete_grid_fit`, both
  Monte Carlo functions and `pick_initial_values`.
- `fit_quality(data, model_curve, ...)` returning `FitQuality`: chi-squared,
  degrees of freedom, R², rms residual, statistics of the normalized
  residuals, overlap per point and overall, best complex scale factor.
- `add_noise(values, x, additive=..., multiplicative=..., seed=...)`: one
  noise model with an additive term (Re/Im) and an exact multiplicative term
  (amplitude/phase), each a number, an array or a function of x;
  `noise_sigma` for the matching standard deviation per point.
- `monte_carlo_noise_fit` (pseudo-measurement study: bias, spread, reported vs
  actual uncertainty, failed fits, fits at a bound; from `initial`, or with
  `random_start=True` from `initial ± initial_std` per run) and
  `monte_carlo_convergence_fit` (same data, random start values: which minima
  are reached), both returning one `MonteCarloResult` (per run: start and
  end values, uncertainties, `at_bound` and `parameters_at_bound` as in
  `FitResult`, cost, chi-squared, R²; summary statistics and the best run), in worker processes, identical for any process count.
- `discrete_grid_fit` (chi-squared on a grid of parameter values, best point
  ready as `initial_values`) and, in `cjcfit.app` (needs the `app` extra),
  `pick_initial_values`: click a start point on the chi-squared map.
- README, AGENTS.md, THEORY.md, six examples with synthetic data in the
  generic axes of cjcplot's examples (among them
  the measured Lab C instrument noise of 2026-06-15 as a reference), tests and
  a GitHub Actions workflow.
