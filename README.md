# cjcfit

[![tests](https://github.com/cjcscheper/cjcfit/actions/workflows/tests.yml/badge.svg)](https://github.com/cjcscheper/cjcfit/actions/workflows/tests.yml)
[![release](https://img.shields.io/github/v/release/cjcscheper/cjcfit)](https://github.com/cjcscheper/cjcfit/releases)

General least-squares fitting: you write the model function, cjcfit fits it
to your data, tells you how good the fit is and how far to trust it, makes
noisy pseudo-measurements, and runs Monte Carlo studies of the fit itself.

- **Your model, by name.** `def model(x, slope, intercept): ...`, fitted with
  one `FitParameter` per argument, like `scipy.optimize.curve_fit`.
- **Bounds, log scaling and priors** per parameter: for quantities that span
  decades (a carrier density, a decay rate) and for prior knowledge from
  elsewhere (a thickness measured with a micrometer).
- **Complex data directly** (a field ratio, an impedance): Re and Im are
  fitted together, with one sigma for both or one for each.
- **Uncertainties, correlations, chi-squared** from the fit; residual
  statistics, R² and more from `fit_quality`.
- **One noise model with two terms**, additive and multiplicative (amplitude
  and phase), each a number, an array or a function of x.
- **Two Monte Carlo studies**: the precision and bias of a fit for a known
  noise (`monte_carlo_noise_fit`), and whether a fit always finds the same
  minimum (`monte_carlo_convergence_fit`); in parallel processes,
  reproducible with a seed.
- **A grid search for the start values** (`discrete_grid_fit`), and a window
  to click one on the chi-squared map (`cjcfit.app.pick_initial_values`).
- **No physics.** No model functions in the package; the examples show some.
- **numpy and scipy** for the core; the window needs matplotlib and
  [cjcplot](https://github.com/cjcscheper/cjcplot) (the `app` extra).

Least squares, chi-squared, the covariance, scaling, priors, complex data,
the noise model and what each Monte Carlo study answers are explained in
[THEORY.md](THEORY.md).

## Install

```bash
pip install git+https://github.com/cjcscheper/cjcfit.git
```

With the start-point window and for the examples:

```bash
pip install "cjcfit[app] @ git+https://github.com/cjcscheper/cjcfit.git"
```

To pin a fixed version, so a script keeps giving the same numbers later on
(see [Releases](https://github.com/cjcscheper/cjcfit/releases) and the
[changelog](CHANGELOG.md)):

```bash
pip install git+https://github.com/cjcscheper/cjcfit.git@v0.1.0
```

For development, clone the repository and install it editable:

```bash
pip install -e ".[test,app]"
```

## Quick start: a straight line

```python
import numpy as np
from cjcfit import FitParameter, add_noise, fit_least_squares

def line(x, slope, intercept):                 # argument names = parameter names
    return slope * x + intercept

x = np.linspace(0.0, 10.0, 15)
data = add_noise(line(x, 2.0, 1.0), additive=0.5, seed=1)   # a pseudo-measurement

fit = fit_least_squares(
    line, x, data,
    parameters=[FitParameter("slope", 1.0), FitParameter("intercept", 0.0)],
    sigma=0.5,                                 # standard deviation of each point
)
fit.values                 # {"slope": 1.925, "intercept": 1.343}
fit.uncertainties          # {"slope": 0.042, "intercept": 0.246}: the textbook formulas
fit.reduced_chi_squared    # 1.04: the model and sigma describe the data
fit.correlation            # (2, 2): slope and intercept are anti-correlated
```

## Parameters

```python
FitParameter(
    "carrier_density", 1e22,          # name and start value (physical units, here m^-3)
    bounds=(1e16, 1e26),              # either side may be None
    scale="log10",                    # the optimiser moves log10(value): steps are factors
    prior_mean=None, prior_std=None,  # Gaussian prior, added as an extra residual
    diff_step=None,                   # absolute finite-difference step, for models with numerical noise
    initial_std=None,                 # spread of random start values in the Monte Carlo studies
)
FitParameter("thickness", 500e-6, scale="normalized", normalization_factor=1e-6)   # moved in µm
```

A `FitParameter` is a frozen dataclass that checks itself when created (a
start value outside the bounds, a log scale with a negative value, half a
prior: all `ValueError`s). `to_scaled(parameter, value)` and
`to_physical(parameter, value)` convert between physical units and the
optimiser's space; `suggest_diff_step(value)` gives a step for `diff_step`.

## What `fit_least_squares` returns

`fit_least_squares(model, x, data, parameters, *, sigma=None, weights=None, initial_values=None, least_squares_options=None)`
returns a frozen `FitResult`, everything in physical units:

| Field                                  | Meaning                                                                     |
|----------------------------------------|-----------------------------------------------------------------------------|
| `values`, `uncertainties`              | best-fit value and one-sigma uncertainty per name                           |
| `covariance`, `correlation`            | `(P, P)`, in the order of `parameter_names`                                 |
| `at_bound`, `parameters_at_bound`      | True if any parameter ended on any bound; which ones (their uncertainty then says nothing) |
| `chi_squared`, `degrees_of_freedom`, `reduced_chi_squared` | of the data residuals with a non-zero weight (Re and Im both count) |
| `r_squared`                            | fraction of the data's variation the model explains (1 = perfect)           |
| `cost`                                 | what was minimised: the weighted sum of squares plus prior terms            |
| `model_curve`, `residuals`, `normalized_residuals` | the model at the optimum, `data - model`, and divided by sigma  |
| `success`, `message`, `n_evaluations`  | from the optimiser                                                          |

With `sigma`, the uncertainties follow from it and `reduced_chi_squared ≈ 1`
checks it. Without `sigma`, every point counts equally and sigma is estimated
from the scatter of the residuals (as `curve_fit` does by default). Complex
`sigma` (`sigma_re + 1j * sigma_im`) weights Re and Im separately; a real
sigma applies to both.

`weights` multiply the residuals the fit minimises (real for both parts, or
`w_re + 1j * w_im`); a weight 0 leaves a point out, e.g. an outlier or a band
outside the usable range:

- **weights alone** are relative `1 / sigma`: only their ratios matter, and
  the uncertainties come from the scatter, as without sigma;
- **sigma and weights** together: the weights are an extra emphasis on top of
  the noise (trust a band less than its sigma says). The uncertainties stay
  those of the given sigma (sandwich covariance, THEORY.md section 4), and
  `chi_squared` is computed without the weights, so it still tests sigma.

`fit_quality(data, model_curve, sigma=..., weights=..., n_parameters=...)`
adds R², residual statistics, an overlap measure and the best scale factor
for any model curve, fitted or not. `discrete_grid_fit` and both Monte Carlo
functions take `sigma` and `weights` too.

## Noise

```python
from cjcfit import add_noise, noise_sigma

noisy = add_noise(
    field_ratio, freq_hz,
    additive=0.01,                           # sigma on Re and on Im (complex: sigma_re + 1j*sigma_im)
    multiplicative=1e-3 + 1j * 0.02,         # 0.1 % amplitude noise, 0.02 rad phase noise
    seed=1,
)
sigma = noise_sigma(field_ratio, freq_hz, additive=0.01, multiplicative=1e-3 + 1j * 0.02)
```

`y_noisy = y (1 + a n1) exp(i b n2) + (s_re n3 + i s_im n4)`, applied exactly.
Each term is a number, an array (one value per point) or a function of x
(`additive=lambda freq_hz: ...`). For real data only the real parts exist.
`noise_sigma` gives the standard deviation per point for the same settings,
ready to pass to a fit as `sigma`. `examples/example_lab_c_noise.py` holds a
measured instrument noise as a reference.

## Monte Carlo

Two different questions:

```python
from cjcfit import monte_carlo_noise_fit, monte_carlo_convergence_fit

# How precise and how biased is the fit for this noise? Fresh noise every run, same start.
result = monte_carlo_noise_fit(
    model, x, true_values={"thickness": 500e-6, "carrier_density": 1e22}, parameters=parameters,
    additive=0.01, multiplicative=1e-3, n_runs=500, seed=1, n_processes=8,
)
result.bias, result.std, result.median_uncertainty, result.n_failed, result.n_at_bound   # runs at any bound

# The same, but every run also starts at a random point initial ± initial_std, as a real fit would.
result = monte_carlo_noise_fit(model, x, true_values, parameters, additive=0.01, random_start=True, n_runs=500, seed=1)

# Does the fit always find the same minimum? Same data every run, start values initial ± initial_std.
result = monte_carlo_convergence_fit(model, x, data, parameters, sigma=sigma, n_runs=200, seed=1, n_processes=8)
result.values, result.cost, result.best_values
```

Both return the same `MonteCarloResult`: per run the start and end values,
uncertainties, `at_bound` (any parameter on any bound, as in `FitResult`) and
`parameters_at_bound`, cost, chi-squared and R²; over the runs the mean,
spread, median reported uncertainty, best run, failed fits and fits at a
bound (`n_at_bound`, and `n_at_bound_per_parameter`). `true_values` and
`bias` exist only for the noise study.

Both run in separate processes (threads would all share one core) and give
the same numbers for any `n_processes`. The model must then be defined at
module level, and the script must start the study under
`if __name__ == "__main__":`.

## Starting values from a grid

```python
from cjcfit import discrete_grid_fit

grid = discrete_grid_fit(model, x, data, {"frequency": np.linspace(0.5, 3, 51), "decay_time": np.logspace(-1, 1, 41)}, sigma=sigma)
fit = fit_least_squares(model, x, data, parameters, sigma=sigma, initial_values=grid.best_values)
```

Or click the start point on the chi-squared map (needs the `app` extra):

```python
from cjcfit.app import pick_initial_values

start = pick_initial_values(model, x, data, {"thickness": ..., "carrier_density": np.logspace(20, 24, 41)})
```

## Examples

Each script in `examples/` makes synthetic data, fits it and saves its
figures next to itself. They plot with
[cjcplot](https://github.com/cjcscheper/cjcplot) and need the `app` extra,
which pins cjcplot to a release tag (now `v0.2.0`).

| Script                                | Shows                                                                   |
|---------------------------------------|-------------------------------------------------------------------------|
| `example_gaussian_peak.py`            | a non-linear fit from a start that is off in position and height        |
| `example_complex_data.py`             | amplitude and delay of `A exp(-i 2 pi x tau)`, fitted in Re and Im       |
| `example_noise_models.py`             | the three forms of a noise term: number, array, function of x           |
| `example_lab_c_noise.py`              | the measured Lab C instrument noise (2026-06-15) as noise terms         |
| `example_grid_fit.py`                 | R² map over two parameters, and the fit started from its best point     |
| `example_monte_carlo.py`              | both studies on two-peak data: where single-peak fits from random starts end, and the spread of the peak position over 5000 noisy fits |

<!-- widths = fig_aspect_ratio[0] x 640 px, so all fonts appear at the same size. -->
<table align="center">
  <tr>
    <td><img src="examples/example_gaussian_peak.png" alt="Gaussian peak: data, fit, true curve and start curve" width="320"></td>
    <td><img src="examples/example_grid_fit_data.png" alt="sine: data, fit from the best grid point and true curve" width="320"></td>
  </tr>
  <tr>
    <td><img src="examples/example_complex_data_amplitude.png" alt="magnitude of a complex fit" width="320"></td>
    <td><img src="examples/example_complex_data_phase.png" alt="phase of a complex fit" width="320"></td>
  </tr>
  <tr>
    <td><img src="examples/example_noise_models_sigma.png" alt="sigma of the three noise forms" width="320"></td>
    <td><img src="examples/example_noise_models_function.png" alt="noisy curve with a bathtub-shaped sigma" width="320"></td>
  </tr>
  <tr>
    <td><img src="examples/example_monte_carlo_data.png" alt="two-peak data and true curve" width="320"></td>
    <td><img src="examples/example_monte_carlo_end_points.png" alt="start and end points of 200 fits" width="320"></td>
  </tr>
  <tr>
    <td><img src="examples/example_monte_carlo_noise.png" alt="histogram of 5000 fitted peak positions" width="320"></td>
    <td><img src="examples/example_lab_c_noise_amplitude.png" alt="field ratio magnitude with Lab C noise" width="320"></td>
  </tr>
</table>
<p align="center">
  <img src="examples/example_grid_fit_map.png" alt="R-squared map over two parameters" width="384">
</p>

## Conventions

- SI units without suffix; other units carry one (`freq_thz`, `time_ps`).
- Residuals are `data - model`; chi-squared counts Re and Im of complex data
  as two values each.
- Arrays of a Monte Carlo result have one entry per run, in run order.

## License

MIT, see [LICENSE](LICENSE).
