# AGENTS.md — cjcfit

Guidance for AI coding agents (and contributors) working in this repository.

This file covers much of the same ground as `README.md`, but for a different
reader: `README.md` is for people who *use* cjcfit (install, a first fit, the
parameters, what the functions return, the examples); this file is for agents
who *work on* it (layout, design rules, conventions, tests, releases).
`THEORY.md` holds the statistics behind it: least squares, the covariance,
scaling, priors, complex data, the noise model and the Monte Carlo studies.
When something changes that more than one of them describes, such as the
public API or a convention, update all of them.

cjcfit is general fitting machinery: least squares with bounds, scaling and
priors, fit quality, a noise model, Monte Carlo studies and a grid search for
start values. It contains **no model functions and no physics**: the user
writes the model (in a script, or a later materials package built on cjctmm)
and passes it in. Model functions appear only in `examples/` and `tests/`, as
illustrations. Do not add transfer matrices, material models, THz specifics
or instrument calibrations to `src/` or `examples/`: measured calibrations
belong to the user's experiments, not to this general package.

## Commands

```bash
pip install -e ".[test]"                         # editable install with pytest (core only, as in CI)
pytest                                           # run all tests; app tests skip without the app extra
pip install -e ".[app]"                          # matplotlib + cjcplot (pinned) for cjcfit.app and examples
python examples/example_gaussian_peak.py         # each example saves its PNG(s) next to itself
```

## Public API

Scripts import only from the package roots, never from deeper submodules:
`cjcfit` for everything that computes, `cjcfit.app` for everything that needs
a screen.

```python
from cjcfit import FitParameter, to_scaled, to_physical, suggest_diff_step
from cjcfit import fit_least_squares, FitResult, fit_quality, FitQuality
from cjcfit import add_noise, noise_sigma
from cjcfit import monte_carlo_noise_fit, monte_carlo_convergence_fit, MonteCarloResult
from cjcfit import discrete_grid_fit, GridFitResult
from cjcfit.app import pick_initial_values
```

## Package layout

```
src/cjcfit/
  __init__.py      core public API and __version__ (its only source); never imports app/
  parameters.py    FitParameter, to_scaled, to_physical, suggest_diff_step; private helpers for
                   bounds, priors, steps and spreads in the scaled space
  quality.py       fit_quality, FitQuality; the one residual definition (sigma split into Re/Im,
                   normalization, Re/Im stacking) used by fitting.py and grid.py
  fitting.py       fit_least_squares, FitResult (scipy least_squares, covariance from the Jacobian)
  noise.py         add_noise, noise_sigma (the two-term noise model)
  monte_carlo.py   monte_carlo_noise_fit, monte_carlo_convergence_fit, their common MonteCarloResult; process pool
  grid.py          discrete_grid_fit, GridFitResult
  app/             needs matplotlib + cjcplot; raises an install hint without them
    __init__.py    dependency check, app public API
    grid_window.py pick_initial_values (the chi-squared map window)
tests/
  test_parameters.py  test_fitting.py  test_weights.py  test_quality.py  test_noise.py  test_monte_carlo.py
  test_grid.py
  test_dependencies.py   core imports no matplotlib or cjcplot; app without the extra gives the install hint
  test_app.py         headless app tests (plt.show replaced, clicks simulated); skipped without the app extra
examples/          one script per topic, PNGs next to it
THEORY.md          the statistics, with references
```

## Design rules

- **No physics, no model functions** in `src/` (see the top of this file).
- **Model functions take named arguments** matching the `FitParameter` names:
  `model(x, **values)`. `x` is passed through untouched; it may be anything.
  Fixed values are the user's business (`functools.partial`, defaults).
- **One residual definition.** Residuals are `data - model`, stacked
  `[Re..., Im...]` (`quality._stack_real_imag`) and multiplied by the factors
  of `quality._residual_scales`: `weight / sigma` for what the optimiser
  minimises, `1 / sigma` (or the weight without sigma) for chi-squared and the
  normalized residuals, and a mask of the points with a non-zero weight.
  `fit_least_squares`, `discrete_grid_fit` and `fit_quality` all use these
  helpers, so chi-squared means the same everywhere. Never write a second copy.
- **Sigma and weights.** Real sigma or weights apply to both Re and Im of
  complex data, complex ones are `re + 1j * im`. Sigma must be positive,
  weights may be 0 (a point left out). With sigma the covariance is absolute
  (with weights too: the sandwich formula); without sigma it is scaled by the
  reduced chi-squared, and weights alone act as relative `1 / sigma`
  (THEORY.md, sections 3 and 4). Chi-squared never contains the weights when
  sigma is given, so it keeps testing sigma.
- **Physical units outside, scaled space inside.** Everything a user gives or
  gets (values, bounds, priors, steps, spreads, uncertainties, covariance) is
  in physical units; only the optimiser sees the scaled space.
- **One noise model.** `add_noise` and `noise_sigma` share `_term_parts`, so
  their reading of a term (number / array / function; real / complex;
  additive real = both parts, multiplicative real = amplitude only) can never
  differ. The four normal arrays are always drawn, in a fixed order.
- **One Monte Carlo result.** Both studies return a `MonteCarloResult`, built
  by one function (`_monte_carlo_result`), with per-run quantities named and
  meant as in `FitResult` (`at_bound` = any parameter on any bound,
  `parameters_at_bound`, `cost`, `chi_squared`, `r_squared`). Only
  `true_values` and `bias` are None without a truth. Keep them aligned when
  `FitResult` changes.
- **Monte Carlo in processes, reproducible.** One `SeedSequence` per study,
  spawned into one seed per run; results must not depend on `n_processes`
  (`test_noise_fit_is_reproducible_and_independent_of_the_process_count`).
  Never switch to threads: a fit holds the interpreter lock.
- **No user-editable settings in the package**: no module-level constants that
  users are expected to change. Every setting is a keyword argument with a
  documented default.
- The grid window keeps its current look (map left with a red +, data solid
  blue and model dashed yellow right, values in the title) until the
  maintainer redesigns it; change its internals only to fix bugs or to reuse
  core functions.

## Coding conventions

- Functions first: everything is written as plain functions (inputs in,
  outputs out). Classes are used ONLY as frozen dataclasses that hold data:
  inputs such as `FitParameter` and results such as `FitResult`,
  `FitQuality`, `GridFitResult`, `MonteCarloResult`. They have no methods (validation in
  `__post_init__` is allowed), no inheritance, no objects that change
  themselves, and there are no classes that only group functions (a module
  does that). A conversion that could be a method, such as to and from the
  scaled space, is a plain function (`to_scaled(parameter, value)`).
- Every logical block inside a function gets a one-line `#` comment saying why
  it exists. Docstrings explain why and how (where a formula comes from, what
  an option changes), with Parameters / Returns / Raises.
- SI units need no suffix; non-SI or scaled values carry one (`freq_thz`,
  `time_ps`, `_mm`). The package itself is unit-agnostic: it never converts.
- Physicist-first names (`fit_least_squares`, `uncertainties`,
  `reduced_chi_squared`, `noise_sigma`), no programmer idioms.
- Wrong inputs raise `ValueError` with a message that says what is wrong and,
  where possible, which parameter.
- Shared code goes in one module and is imported, never copied.
- Remove unused code instead of leaving it commented out.
- Python 3.10 compatible (no syntax newer than 3.10).

## Dependencies

- All dependencies live in `pyproject.toml`, the only list: `[project]
  dependencies` (numpy, scipy), the `test` extra (pytest) and the `app` extra
  (matplotlib, cjcplot pinned to a release tag). There is no `requirements.txt`.
- The core (everything outside `app/`) and `tests/` never import matplotlib or
  cjcplot (`test_core_does_not_import_matplotlib_or_cjcplot` checks it); only
  `app/` and `examples/` may, via the pinned `app` extra. When upgrading the
  pin, change the tag in pyproject.toml (and in the README) and regenerate all
  example PNGs. Examples use cjcplot's public API only (plus the axes it
  returns), save their PNGs with `save_path=..., fig_show=False`, give every
  figure its own file (no subplots), size 1D figures
  `fig_aspect_ratio=(0.5, 0.5)` and 2D maps `(0.6, 0.5)`, use
  `plot_histogram_1d` for histograms, and draw data as blue markers with
  error bars, fits as yellow dashed lines (`"--"`) and true (noise-free)
  curves as red solid lines below the fit (trace order data, fit, true, so
  with `first_on_top` the fit lies on the true curve); further curves (a
  start, a local minimum, a stuck fit) grey dotted (`":"`), last. A pseudo-measurement whose
  fit is meant to match the truth uses noise the fit recovers without bias
  (additive noise, or small phase noise; THEORY.md, section 8).
- Examples that run Monte Carlo studies with `n_processes > 1` define their
  model at module level and run under `if __name__ == "__main__":`.

## Tests

- Tests check values that can be verified by hand: the textbook
  sigma_slope and sigma_intercept of a straight line, sigma_A = sigma/sqrt(N)
  and sigma_tau of a complex delay, the inverse-variance mean of data and a
  prior, chi-squared = 4N for an offset of 2 sigma, the two minima +-2 of
  `root^2 x`. Statistical tests use a fixed seed and a tolerance of several
  times the statistical spread, stated in a comment.
- Wrong inputs raise `ValueError`s with clear messages, and those messages are
  tested.
- New or changed functionality comes with new or updated tests in the same
  change, and `pytest` passes before committing. When the public API changes,
  update the README, this file, THEORY.md, CHANGELOG.md and the affected
  examples (and their PNGs) too.
- Every test has a one-line docstring saying what it checks; non-obvious lines
  (the reason behind a value or tolerance) get a short comment.
- Model functions used with `n_processes > 1` in tests are defined at module
  level of the test file.
- App tests run headless (Agg, `plt.show` replaced with `monkeypatch`, clicks
  sent through the canvas callbacks) and skip when the app extra is missing.
- GitHub Actions (`.github/workflows/tests.yml`) runs `pytest` on Python 3.10
  and 3.14 for every push and pull request, core only (without the app extra).

## Changelog and releases

- Every change that users notice (new or changed functionality, fixes, docs
  people rely on) gets a line in `CHANGELOG.md` under `## [Unreleased]`, in the
  same change, sorted under `### Added`, `### Changed`, `### Fixed` or `### Removed`.
- Before finishing a change, check that `CHANGELOG.md` describes it.
- Versions follow MAJOR.MINOR.PATCH: patch for fixes, minor for new features,
  major for changes that break existing scripts (below 1.0 the API may still change).
  A change of a convention, of the noise model or of returned values counts as breaking.
- A release:
  1. rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty
     `## [Unreleased]` above it;
  2. set the new version in `src/cjcfit/__init__.py` (`__version__`), its only
     source: `pyproject.toml` reads it from there; also update the pinned
     version in the README's install command;
  3. commit, tag `vX.Y.Z`, push the commit and the tag, and publish the GitHub
     release with the changelog section as its notes.

## Git workflow

- Do not commit, push, tag, merge or force-push without the maintainer's approval.
- Keep each change focused: no unrelated reformatting or renaming mixed in.
- Keep generated files in the same change as the code that produces them:
  regenerate an example's PNGs when the example or the code it uses changes.
- In the commit message (or pull request), summarise what changes for users and
  how it was checked (tests run, examples regenerated).
