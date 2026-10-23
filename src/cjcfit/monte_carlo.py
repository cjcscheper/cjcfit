"""Monte Carlo studies: many fits, to learn about the fit rather than about one measurement.

Two different questions, two functions, one result (THEORY.md, section 9):

- :func:`monte_carlo_noise_fit` -- *How precise and how biased is the result
  for a known noise level?* Every run fits a fresh pseudo-measurement (the
  model at the true values plus new noise), from the same start values or,
  with ``random_start=True``, from random ones. The spread of the results is
  the real uncertainty, their mean minus the truth the bias; comparing the
  spread with the uncertainty each fit reports tests the linearised covariance.
- :func:`monte_carlo_convergence_fit` -- *Does the fit always find the same
  minimum?* Every run fits the *same* data from different start values. A
  spread here is not an uncertainty: it shows local minima, flat valleys and
  start values the fit cannot recover from.

Both return a :class:`MonteCarloResult`: the same per-run arrays and the same
statistics, so plots and checks work on either. Only ``true_values`` and
``bias`` need a truth and are None for a convergence study.

Both run the fits in separate processes (``n_processes``): a fit is pure
Python and holds the interpreter lock, so threads would all share one core.
Each run draws its random numbers from its own seed, spawned from ``seed``,
so the results are identical for any number of processes. With
``n_processes > 1`` the model must be picklable, i.e. a function defined at
module level (or a ``functools.partial`` of one), not a lambda or a function
defined inside another function, and a script must start the study under
``if __name__ == "__main__":`` (on Windows and macOS each process imports
the script again).
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Mapping, Sequence

import numpy as np

from cjcfit.fitting import _checked_parameters, fit_least_squares
from cjcfit.noise import add_noise
from cjcfit.parameters import FitParameter, _scaled_bounds, _scaled_std, to_physical, to_scaled
from cjcfit.quality import _matching_data_and_model

# What every run of a study shares, set once per worker process (see _set_shared).
_SHARED: tuple | None = None


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of :func:`monte_carlo_noise_fit` and :func:`monte_carlo_convergence_fit`.

    Arrays have one entry per run, in run order. Each run is one
    :func:`cjcfit.fit_least_squares`; its numbers have the same meaning as in a
    :class:`cjcfit.FitResult`. Statistics are taken over the successful runs.

    Attributes
    ----------
    parameter_names : tuple[str, ...]
    true_values : dict[str, float] or None
        The values the pseudo-measurements were made with (noise study); None
        for a convergence study, which fits given data.
    start_values : dict[str, ndarray]
        Where each run started.
    values, uncertainties : dict[str, ndarray]
        Fitted value and reported one-sigma uncertainty of every run (NaN for
        a run that raised an error).
    at_bound : ndarray of bool
        Per run, as :attr:`cjcfit.FitResult.at_bound`: True when any parameter
        ended on any of its bounds.
    parameters_at_bound : tuple[tuple[str, ...], ...]
        Per run, which parameters ended on a bound (``()`` if none).
    success : ndarray of bool
        Whether the optimiser converged (False also for runs that raised an error).
    cost, chi_squared, r_squared : ndarray
        Of every run's fit, as in :class:`cjcfit.FitResult`. Runs that end with
        the same lowest cost found the same, presumably global, minimum; a low
        R² flags a poor fit.
    n_evaluations : ndarray of int
        Model evaluations per run.
    mean, std : dict[str, float]
        Mean and standard deviation (ddof = 1) of the fitted values. In a noise
        study ``std`` is the actual uncertainty of the fit; in a convergence
        study it is the spread of the end points, not an uncertainty.
    bias : dict[str, float] or None
        ``mean - true value`` (noise study); None for a convergence study.
    median_uncertainty : dict[str, float]
        Median of the reported uncertainties; close to ``std`` in a noise study
        when the linearised covariance is trustworthy.
    best_values : dict[str, float]
        End point of the run with the lowest cost.
    best_cost : float
    n_failed : int
        Runs that did not converge or raised an error.
    n_at_bound : int
        Successful runs that ended on any bound (``at_bound`` True).
    n_at_bound_per_parameter : dict[str, int]
        Successful runs that ended on a bound of that parameter.
    seed : int
        The seed of the study; passing it again repeats every run exactly.
    first_error : str or None
        Message of the first error raised in a run, if any.
    """

    parameter_names: tuple[str, ...]
    true_values: dict[str, float] | None
    start_values: dict[str, np.ndarray]
    values: dict[str, np.ndarray]
    uncertainties: dict[str, np.ndarray]
    at_bound: np.ndarray
    parameters_at_bound: tuple[tuple[str, ...], ...]
    success: np.ndarray
    cost: np.ndarray
    chi_squared: np.ndarray
    r_squared: np.ndarray
    n_evaluations: np.ndarray
    mean: dict[str, float]
    std: dict[str, float]
    bias: dict[str, float] | None
    median_uncertainty: dict[str, float]
    best_values: dict[str, float]
    best_cost: float
    n_failed: int
    n_at_bound: int
    n_at_bound_per_parameter: dict[str, int]
    seed: int
    first_error: str | None


def monte_carlo_noise_fit(
    model: Callable,
    x,
    true_values: Mapping[str, float],
    parameters: Sequence[FitParameter],
    *,
    additive=0.0,
    multiplicative=0.0,
    sigma=None,
    weights=None,
    random_start: bool = False,
    n_runs: int = 100,
    seed: int | None = None,
    n_processes: int = 1,
    show_progress: bool = False,
    least_squares_options: Mapping[str, object] | None = None,
) -> MonteCarloResult:
    """Pseudo-measurement study: fit ``n_runs`` noisy copies of the model at ``true_values``.

    Answers: for this noise, how far from the truth does the fit land (bias),
    how much does it scatter (std), and is the uncertainty each fit reports
    right (median_uncertainty vs std)?

    By default every run fits from the parameters' ``initial`` values, so the
    result describes the noise alone. With ``random_start=True`` every run
    also starts somewhere else, at ``initial ± initial_std`` (drawn exactly as
    in :func:`monte_carlo_convergence_fit`), as a real measurement would be
    fitted from a start that is only roughly right: runs that end in another
    minimum then show up in the spread, the bias and ``n_failed``. Comparing
    the two shows how much of the scatter the start values add.

    Parameters
    ----------
    model : callable
        ``model(x, **values)`` as for :func:`cjcfit.fit_least_squares`;
        picklable (defined at module level) when ``n_processes > 1``.
    x : anything
        Passed to the model and to noise terms that are functions.
    true_values : mapping
        The true value of every fitted parameter, by name.
    parameters : sequence of FitParameter
        As for :func:`cjcfit.fit_least_squares`.
    additive, multiplicative : number, array-like or callable
        The noise of each pseudo-measurement, exactly as in :func:`cjcfit.add_noise`.
    sigma : float or array-like, optional
        Passed to every fit. ``noise_sigma(model(x, **true_values), x,
        additive=..., multiplicative=...)`` weights the points as the noise
        does, and makes the reported uncertainties follow from it. None: an
        unweighted fit, uncertainties from the scatter of the residuals.
    weights : float or array-like, optional
        Passed to every fit, as for :func:`cjcfit.fit_least_squares`.
    random_start : bool, default False
        Start each run at ``initial ± initial_std`` instead of at ``initial``
        (parameters without ``initial_std`` still start at ``initial``). The
        noise of each run is the same with or without it.
    n_runs : int, default 100
        Number of pseudo-measurements.
    seed : int, optional
        Seed of the whole study; None picks one (stored in the result).
    n_processes : int, default 1
        Worker processes (capped at ``n_runs``); the results do not depend on it.
    show_progress : bool, default False
        Print ``runs done / n_runs`` on one line of the terminal (stderr).
    least_squares_options : mapping, optional
        As for :func:`cjcfit.fit_least_squares`.

    Returns
    -------
    MonteCarloResult
    """
    # The truth must name exactly the fitted parameters, and the model must work there.
    parameters = _checked_parameters(parameters)
    names = tuple(parameter.name for parameter in parameters)
    if set(true_values) != set(names):
        raise ValueError(f"true_values must give exactly the fitted parameters {list(names)}, got {list(true_values)}.")
    true_values = {name: float(true_values[name]) for name in names}
    true_curve = np.asarray(model(x, **true_values))
    _, true_curve = _matching_data_and_model(true_curve, true_curve)
    if not np.all(np.isfinite(true_curve)):
        raise ValueError(f"The model returns NaN or inf at the true values {true_values}.")

    # One seed per run, spawned from the study's seed: the same runs for any number of processes.
    seed_sequence = np.random.SeedSequence(seed)
    run_seeds = seed_sequence.spawn(_checked_count(n_runs, "n_runs"))

    # Start values: initial, or drawn from a second set of seeds, so the noise of each run does not change with them.
    if random_start:
        starts = [_random_start(parameters, np.random.default_rng(s)) for s in seed_sequence.spawn(len(run_seeds))]
    else:
        starts = [{parameter.name: float(parameter.initial) for parameter in parameters}] * len(run_seeds)

    # The runs, in order, gathered into the common result.
    shared = (model, x, true_curve, parameters, additive, multiplicative, sigma, weights, least_squares_options)
    items = list(zip(run_seeds, starts))
    outcomes = _run_all(_noise_run, shared, items, n_processes, show_progress, "Monte Carlo noise fit")
    return _monte_carlo_result(names, starts, outcomes, true_values, int(seed_sequence.entropy))


def monte_carlo_convergence_fit(
    model: Callable,
    x,
    data,
    parameters: Sequence[FitParameter],
    *,
    sigma=None,
    weights=None,
    n_runs: int = 100,
    seed: int | None = None,
    n_processes: int = 1,
    show_progress: bool = False,
    least_squares_options: Mapping[str, object] | None = None,
) -> MonteCarloResult:
    """Robustness study: fit the same ``data`` ``n_runs`` times from random start values.

    Each run starts at ``initial ± initial_std`` per parameter (normal; for log
    scales log-normal with the same spread to first order; clipped into the
    bounds; parameters without ``initial_std`` always start at ``initial``).
    If all runs end at the same values with the same cost, the fit finds its
    minimum from anywhere in that range; several groups of end points mean
    several minima, and the one with the lowest cost (``best_values``) is the
    best fit. The spread of the end points is *not* an uncertainty of the
    result (the data never change); use :func:`monte_carlo_noise_fit` or the
    fit's own uncertainties for that. ``true_values`` and ``bias`` of the
    result are None: there is no truth here.

    Parameters
    ----------
    model, x, data, parameters, sigma, weights, least_squares_options
        As for :func:`cjcfit.fit_least_squares`.
    n_runs, seed, n_processes, show_progress
        As for :func:`monte_carlo_noise_fit`.

    Returns
    -------
    MonteCarloResult
    """
    # Same checks as a single fit; one seed per run for its start values.
    parameters = _checked_parameters(parameters)
    names = tuple(parameter.name for parameter in parameters)
    seed_sequence = np.random.SeedSequence(seed)
    run_seeds = seed_sequence.spawn(_checked_count(n_runs, "n_runs"))

    # The start values of every run, drawn here so they are known even for runs that fail.
    starts = [_random_start(parameters, np.random.default_rng(run_seed)) for run_seed in run_seeds]

    # The runs, in order, gathered into the common result.
    shared = (model, x, np.asarray(data), parameters, sigma, weights, least_squares_options)
    outcomes = _run_all(_convergence_run, shared, starts, n_processes, show_progress, "Monte Carlo convergence fit")
    return _monte_carlo_result(names, starts, outcomes, None, int(seed_sequence.entropy))


def _monte_carlo_result(names, starts, outcomes, true_values, seed) -> MonteCarloResult:
    """Per-run outcomes -> the common :class:`MonteCarloResult` with its statistics over the successful runs."""
    # One array per quantity; statistics only over runs that converged (a failed fit says nothing about precision).
    columns = _collect(outcomes, names)
    ok = columns["success"]
    values = columns["values"]
    mean = {name: _mean(values[name][ok]) for name in names}

    # The best run: lowest cost among those that converged.
    if np.any(ok):
        best = int(np.flatnonzero(ok)[np.argmin(columns["cost"][ok])])
        best_values = {name: float(values[name][best]) for name in names}
        best_cost = float(columns["cost"][best])
    else:
        best_values, best_cost = dict.fromkeys(names, np.nan), np.nan

    return MonteCarloResult(
        parameter_names=names,
        true_values=true_values,
        start_values={name: np.array([start[name] for start in starts]) for name in names},
        values=values,
        uncertainties=columns["uncertainties"],
        at_bound=columns["at_bound"],
        parameters_at_bound=columns["parameters_at_bound"],
        success=ok,
        cost=columns["cost"],
        chi_squared=columns["chi_squared"],
        r_squared=columns["r_squared"],
        n_evaluations=columns["n_evaluations"],
        mean=mean,
        std={name: _std(values[name][ok]) for name in names},
        bias=None if true_values is None else {name: mean[name] - true_values[name] for name in names},
        median_uncertainty={name: _median(columns["uncertainties"][name][ok]) for name in names},
        best_values=best_values,
        best_cost=best_cost,
        n_failed=int(np.sum(~ok)),
        n_at_bound=int(np.sum(columns["at_bound"][ok])),
        n_at_bound_per_parameter={
            name: sum(1 for flag, at in zip(ok, columns["parameters_at_bound"]) if flag and name in at)
            for name in names
        },
        seed=seed,
        first_error=columns["first_error"],
    )


def _checked_count(count: int, name: str) -> int:
    """A positive whole number of runs or processes."""
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"{name} must be a positive integer, got {count!r}.")
    return int(count)


def _random_start(parameters: Sequence[FitParameter], generator: np.random.Generator) -> dict[str, float]:
    """Start values ``initial ± initial_std``, drawn in the scaled space (log-normal for log scales), inside the bounds."""
    # One normal number per parameter, always drawn, so a parameter's start does not depend on the others' settings.
    start = {}
    for parameter, normal in zip(parameters, generator.standard_normal(len(parameters))):
        spread = parameter.initial_std or 0.0
        scaled = to_scaled(parameter, parameter.initial) + _scaled_std(parameter, spread, parameter.initial) * normal
        start[parameter.name] = float(to_physical(parameter, np.clip(scaled, *_scaled_bounds(parameter))))
    return start


def _noise_run(shared: tuple, item: tuple[np.random.SeedSequence, dict[str, float]]) -> dict:
    """One pseudo-measurement: true curve + fresh noise, fitted from the given start values."""
    model, x, true_curve, parameters, additive, multiplicative, sigma, weights, options = shared
    run_seed, start = item
    data = add_noise(true_curve, x, additive=additive, multiplicative=multiplicative, seed=np.random.default_rng(run_seed))
    return _fit_outcome(model, x, data, parameters, sigma, weights, options, start)


def _convergence_run(shared: tuple, start: dict[str, float]) -> dict:
    """One fit of the shared data from the given start values."""
    model, x, data, parameters, sigma, weights, options = shared
    return _fit_outcome(model, x, data, parameters, sigma, weights, options, start)


def _fit_outcome(model, x, data, parameters, sigma, weights, options, start) -> dict:
    """The numbers of one fit that a study keeps; an error in the run becomes a failed run with its message."""
    # A model may fail at odd parameter values in a few runs; that should not end the whole study.
    # A run without a covariance reports inf uncertainties; its warning, once per run, would only clutter the terminal.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The covariance could not be estimated", category=RuntimeWarning)
            fit = fit_least_squares(
                model, x, data, parameters, sigma=sigma, weights=weights, initial_values=start,
                least_squares_options=options,
            )
    except Exception as error:  # noqa: BLE001 -- any error of the user's model counts as a failed run
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "values": fit.values,
        "uncertainties": fit.uncertainties,
        "parameters_at_bound": fit.parameters_at_bound,
        "success": fit.success,
        "cost": fit.cost,
        "chi_squared": fit.chi_squared,
        "r_squared": fit.r_squared,
        "n_evaluations": fit.n_evaluations,
    }


def _set_shared(shared: tuple) -> None:
    """Worker-process initializer: keep what all runs share, so it is sent once per process, not per run."""
    global _SHARED
    _SHARED = shared


def _run_with_shared(run_function: Callable, item) -> dict:
    """Run one item in a worker process with the shared data set by :func:`_set_shared`."""
    return run_function(_SHARED, item)


def _run_all(run_function: Callable, shared: tuple, items: list, n_processes: int, show_progress: bool, label: str) -> list:
    """``run_function(shared, item)`` for every item, in order, in this process or in a process pool."""
    # More processes than runs would only start idle workers.
    n_processes = min(_checked_count(n_processes, "n_processes"), len(items))
    outcomes = []

    def report(done: int) -> None:
        """One self-overwriting progress line, updated at every whole percent; a newline at the end."""
        if show_progress and (done == len(items) or done * 100 // len(items) != (done - 1) * 100 // len(items)):
            print(f"\r{label}: {done}/{len(items)} runs", end="\n" if done == len(items) else "", file=sys.stderr, flush=True)

    # One process: plain loop, no pickling.
    if n_processes == 1:
        for item in items:
            outcomes.append(run_function(shared, item))
            report(len(outcomes))
        return outcomes

    # Several processes: map keeps the run order; a few chunks per process balance the load.
    chunk_size = max(1, len(items) // (4 * n_processes))
    with ProcessPoolExecutor(max_workers=n_processes, initializer=_set_shared, initargs=(shared,)) as pool:
        for outcome in pool.map(partial(_run_with_shared, run_function), items, chunksize=chunk_size):
            outcomes.append(outcome)
            report(len(outcomes))
    return outcomes


def _collect(outcomes: list[dict], names: tuple[str, ...]) -> dict:
    """Per-run outcomes -> one array per quantity (NaN / False for runs that raised an error)."""
    # Runs that raised an error have only a message.
    failed = [outcome for outcome in outcomes if "error" in outcome]
    nan_values = dict.fromkeys(names, np.nan)
    filled = [
        outcome if "error" not in outcome else {
            "values": nan_values, "uncertainties": nan_values, "parameters_at_bound": (),
            "success": False, "cost": np.nan, "chi_squared": np.nan, "r_squared": np.nan, "n_evaluations": 0,
        }
        for outcome in outcomes
    ]

    # Columns per quantity, and per parameter for the per-parameter ones; at_bound is the OR over the parameters.
    return {
        "values": {name: np.array([o["values"][name] for o in filled]) for name in names},
        "uncertainties": {name: np.array([o["uncertainties"][name] for o in filled]) for name in names},
        "parameters_at_bound": tuple(tuple(o["parameters_at_bound"]) for o in filled),
        "at_bound": np.array([bool(o["parameters_at_bound"]) for o in filled], dtype=bool),
        "success": np.array([o["success"] for o in filled], dtype=bool) & np.array(
            [np.all(np.isfinite(list(o["values"].values()))) for o in filled]
        ),
        "cost": np.array([o["cost"] for o in filled], dtype=float),
        "chi_squared": np.array([o["chi_squared"] for o in filled], dtype=float),
        "r_squared": np.array([o["r_squared"] for o in filled], dtype=float),
        "n_evaluations": np.array([o["n_evaluations"] for o in filled], dtype=int),
        "first_error": failed[0]["error"] if failed else None,
    }


def _mean(values: np.ndarray) -> float:
    """Mean, NaN for no values (instead of numpy's warning)."""
    return float(np.mean(values)) if values.size else np.nan


def _std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof = 1), NaN for fewer than two values."""
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _median(values: np.ndarray) -> float:
    """Median, NaN for no values."""
    return float(np.median(values)) if values.size else np.nan
