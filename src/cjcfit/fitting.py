"""Least-squares fit of a user's model function to real or complex data.

:func:`fit_least_squares` works like ``scipy.optimize.curve_fit``, with the
parameters described by :class:`cjcfit.FitParameter` (bounds, scaling,
priors, finite-difference steps) and passed to the model by name:

    def model(x, slope, intercept):
        return slope * x + intercept

The optimiser is ``scipy.optimize.least_squares`` (trust-region reflective,
which handles bounds). The uncertainties come from the Jacobian at the
optimum (THEORY.md, section 4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from cjcfit.parameters import (
    FitParameter,
    _physical_per_scaled,
    _scaled_bounds,
    _scaled_diff_step,
    _scaled_prior,
    to_physical,
    to_scaled,
)
from cjcfit.quality import _matching_data_and_model, _residual_scales, _stack_real_imag, fit_quality

# Residual given to the optimiser where the model returns NaN or inf (e.g. outside its valid range):
# large enough that the optimiser steps back, finite so it can still compute with it.
_LARGE_RESIDUAL = 1e100

# Options of scipy's least_squares that fit_least_squares sets itself from the FitParameters.
_RESERVED_OPTIONS = ("x0", "bounds", "diff_step", "args", "kwargs")


@dataclass(frozen=True)
class FitResult:
    """Result of :func:`fit_least_squares`, everything in physical units.

    Attributes
    ----------
    parameter_names : tuple[str, ...]
        Names in the order of the ``parameters`` given; the order of the rows
        and columns of ``covariance`` and ``correlation``.
    values : dict[str, float]
        Best-fit value per parameter.
    uncertainties : dict[str, float]
        One-sigma uncertainty per parameter, ``sqrt`` of the covariance
        diagonal. ``inf`` when the data do not determine all parameters.
    covariance, correlation : ndarray, shape (P, P)
        Linearised covariance of the parameters (from the Jacobian at the
        optimum, THEORY.md, section 4) and the correlation coefficients
        ``cov_ij / (sigma_i sigma_j)``: near ±1 means two parameters can
        compensate each other.
    at_bound : bool
        True when any parameter ended on any of its bounds (lower or upper).
        The uncertainty of such a parameter says nothing: the true optimum
        may lie beyond the bound.
    parameters_at_bound : tuple[str, ...]
        Names of the parameters that ended on a bound (empty if none), to see
        which one when ``at_bound`` is True.
    success : bool
        Whether the optimiser reported convergence.
    message : str
        The optimiser's reason for stopping.
    cost : float
        Sum of all squared residuals the optimiser minimised: ``chi_squared``
        plus the prior terms. (Twice scipy's ``cost``, which has a factor 1/2.)
    chi_squared : float
        Sum of the squared normalized data residuals (Re and Im both count),
        over the points with a non-zero weight. With sigma it does not contain
        the weights, so ``reduced_chi_squared`` still tests sigma.
    degrees_of_freedom : int
        Real data values with a non-zero weight (2 per complex point) minus
        fitted parameters.
    reduced_chi_squared : float
        ``chi_squared / degrees_of_freedom``; about 1 for a good model with the
        right sigma (THEORY.md, section 3). Only meaningful with sigma.
    r_squared : float
        ``1 - sum(residual²) / sum((data - mean)²)`` over the real values with
        a non-zero weight (Re and Im each around their own mean): the fraction
        of the data's variation the model explains, 1 for a perfect fit.
        Without sigma or weights inside, so it compares fits with different
        sigma; as :attr:`FitQuality.r_squared`.
    n_evaluations : int
        Number of model evaluations (without those for the Jacobian).
    model_curve : ndarray
        The model at the best-fit values; shape and type of the data.
    residuals : ndarray
        ``data - model_curve``.
    normalized_residuals : ndarray
        Residuals divided by sigma (Re and Im separately); without sigma but
        with weights, times the weights; with neither equal to ``residuals``.
    """

    parameter_names: tuple[str, ...]
    values: dict[str, float]
    uncertainties: dict[str, float]
    covariance: np.ndarray
    correlation: np.ndarray
    at_bound: bool
    parameters_at_bound: tuple[str, ...]
    success: bool
    message: str
    cost: float
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    r_squared: float
    n_evaluations: int
    model_curve: np.ndarray
    residuals: np.ndarray
    normalized_residuals: np.ndarray


def fit_least_squares(
    model: Callable,
    x,
    data,
    parameters: Sequence[FitParameter],
    *,
    sigma=None,
    weights=None,
    initial_values: Mapping[str, float] | None = None,
    least_squares_options: Mapping[str, object] | None = None,
) -> FitResult:
    """Fit ``model(x, **values)`` to ``data`` by least squares.

    Parameters
    ----------
    model : callable
        ``model(x, name_1=..., name_2=...)`` with one keyword argument per
        :class:`FitParameter` (by its ``name``), returning an array of the
        data's shape, real or complex. Values the fit should not change are
        set with ``functools.partial`` or a default argument.
    x : anything
        Passed to the model unchanged (frequencies, times, a tuple of axes...).
    data : array-like
        The measurement, real or complex, any shape. Complex data are fitted
        in Re and Im at the same time (THEORY.md, section 7).
    parameters : sequence of FitParameter
        What is fitted, with start values, bounds, scales and priors.
    sigma : float or array-like, optional
        Standard deviation of each data point (a number for all, or one per
        point). Real sigma applies to both Re and Im of complex data; complex
        sigma is ``sigma_re + 1j * sigma_im``. With sigma the residuals are
        divided by it and the covariance is taken as is: the uncertainties
        follow from the sigma given, and ``reduced_chi_squared`` tests it.
        Without sigma every point counts equally and the covariance is scaled
        by ``chi_squared / degrees_of_freedom``, i.e. sigma is estimated from
        the scatter of the residuals (as ``curve_fit`` does by default); give
        sigma whenever a parameter has a prior, which is in physical units.
    weights : float or array-like, optional
        Factor per data point (or per part: real weights apply to Re and Im,
        complex weights are ``w_re + 1j * w_im``) that multiplies the residual
        the optimiser minimises; 0 leaves a point out. Without sigma they are
        relative ``1 / sigma`` (only their ratios matter): the fit is weighted
        and the uncertainties are scaled by the scatter, as without sigma.
        With sigma they are an extra emphasis on top of it (e.g. lowering a
        band you trust less than its sigma says): the covariance then uses the
        sandwich formula, so the uncertainties stay those of the sigma given
        (THEORY.md, section 4), and chi-squared is computed without them.
    initial_values : mapping, optional
        Start values by name that replace the parameters' ``initial``, e.g.
        ``discrete_grid_fit(...).best_values``. Clipped into the bounds.
    least_squares_options : mapping, optional
        Extra options for ``scipy.optimize.least_squares``, e.g.
        ``{"max_nfev": 400, "jac": "3-point", "x_tol": 1e-10}``. Bounds and
        steps come from the parameters and cannot be given here. The
        statistics assume the default ``loss="linear"``.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        For duplicate or unknown parameter names, a model output that does
        not match the data, a model that is not finite at the start values,
        or an invalid sigma.
    """
    # Parameters, data and sigma checked once, before the optimiser calls the model many times.
    parameters = _checked_parameters(parameters)
    data = np.asarray(data)
    data, _ = _matching_data_and_model(data, data)
    scales = _residual_scales(data, sigma, weights)
    options = dict(least_squares_options or {})
    reserved = [key for key in _RESERVED_OPTIONS if key in options]
    if reserved:
        raise ValueError(f"least_squares_options may not contain {reserved}: they are set from the FitParameters.")

    # Start in the scaled space, inside the bounds.
    start_physical = _start_values(parameters, initial_values)
    lower, upper = np.array([_scaled_bounds(parameter) for parameter in parameters]).T
    start_scaled = np.clip([to_scaled(p, start_physical[p.name]) for p in parameters], lower, upper)

    # A model that fails at the start would leave the optimiser with nothing to go on: say so directly.
    start_curve = _evaluate_model(model, x, parameters, start_scaled)
    _, start_curve = _matching_data_and_model(data, start_curve)
    if not np.all(np.isfinite(start_curve)):
        raise ValueError(f"The model returns NaN or inf at the start values {_values_dict(parameters, start_scaled)}.")

    def residual_vector(scaled_values: np.ndarray) -> np.ndarray:
        """Normalized data residuals, Re and Im stacked, followed by one residual per prior."""
        return _residual_vector(model, x, data, scales, parameters, scaled_values)

    # Finite-difference steps: only where a parameter asks for one, converted to scipy's relative steps.
    diff_step = _relative_diff_steps(parameters, start_physical, start_scaled, options.get("jac", "2-point"))
    if diff_step is not None:
        options["diff_step"] = diff_step

    # The optimisation itself, in the scaled space.
    solution = least_squares(residual_vector, start_scaled, bounds=(lower, upper), **options)
    best_scaled = solution.x

    # Statistics of the data residuals at the optimum, from fit_quality, so a fit and a later
    # fit_quality of its model curve always agree (priors are not data and do not count; nor do weights 0).
    best_curve = _evaluate_model(model, x, parameters, best_scaled)
    _, best_curve = _matching_data_and_model(data, best_curve)
    quality = fit_quality(data, best_curve, sigma=sigma, weights=weights, n_parameters=len(parameters))

    # Covariance in the scaled space: absolute with sigma, scaled by the scatter without (THEORY.md, section 4).
    covariance_scaled = _covariance_from_jacobian(solution.jac)
    if not scales.has_sigma:
        covariance_scaled = covariance_scaled * quality.reduced_chi_squared
    elif weights is not None and np.all(np.isfinite(covariance_scaled)):
        # Sigma and weights: the weighted residuals have variance weight^2 (priors 1), hence the sandwich H^-1 M H^-1.
        row_variance = np.concatenate([scales.weights**2, np.ones(solution.jac.shape[0] - scales.weights.size)])
        middle = (solution.jac.T * row_variance) @ solution.jac
        covariance_scaled = covariance_scaled @ middle @ covariance_scaled

    # To physical units: multiply by d(physical)/d(scaled) on both sides.
    best_values = _values_dict(parameters, best_scaled)
    derivative = np.array([_physical_per_scaled(p, best_values[p.name]) for p in parameters])
    covariance = derivative[:, None] * covariance_scaled * derivative[None, :]
    uncertainties = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(uncertainties, uncertainties)

    # A parameter is at a bound (lower or upper) when the optimiser says so or when it sits there to rounding.
    on_bound = (solution.active_mask != 0) | np.isclose(best_scaled, lower) | np.isclose(best_scaled, upper)
    parameters_at_bound = tuple(p.name for p, flag in zip(parameters, on_bound) if flag)

    return FitResult(
        parameter_names=tuple(p.name for p in parameters),
        values=best_values,
        uncertainties={p.name: float(u) for p, u in zip(parameters, uncertainties)},
        covariance=covariance,
        correlation=correlation,
        at_bound=bool(parameters_at_bound),
        parameters_at_bound=parameters_at_bound,
        success=bool(solution.success),
        message=str(solution.message),
        cost=float(2.0 * solution.cost),
        chi_squared=quality.chi_squared,
        degrees_of_freedom=quality.degrees_of_freedom,
        reduced_chi_squared=quality.reduced_chi_squared,
        r_squared=quality.r_squared,
        n_evaluations=int(solution.nfev),
        model_curve=best_curve,
        residuals=quality.residuals,
        normalized_residuals=quality.normalized_residuals,
    )


def _checked_parameters(parameters: Iterable[FitParameter]) -> tuple[FitParameter, ...]:
    """The parameters as a tuple: at least one, all FitParameters, no name twice."""
    # Every later step indexes by name, so names must be unique.
    checked = tuple(parameters)
    if not checked:
        raise ValueError("Give at least one FitParameter.")
    if not all(isinstance(parameter, FitParameter) for parameter in checked):
        raise ValueError("parameters must be FitParameter objects.")
    names = [parameter.name for parameter in checked]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Parameter names must be unique; repeated: {duplicates}.")
    return checked


def _start_values(parameters: Sequence[FitParameter], initial_values: Mapping[str, float] | None) -> dict[str, float]:
    """Start value per name: the parameter's ``initial`` unless ``initial_values`` replaces it."""
    # Unknown names are most likely typos; refuse them instead of ignoring them.
    start = {parameter.name: float(parameter.initial) for parameter in parameters}
    if initial_values is not None:
        unknown = sorted(set(initial_values) - set(start))
        if unknown:
            raise ValueError(f"initial_values has names that are not fitted: {unknown}; fitted are {list(start)}.")
        start.update({name: float(value) for name, value in initial_values.items()})
    return start


def _values_dict(parameters: Sequence[FitParameter], scaled_values) -> dict[str, float]:
    """Scaled values -> {name: physical value}."""
    return {p.name: float(to_physical(p, value)) for p, value in zip(parameters, scaled_values)}


def _evaluate_model(model: Callable, x, parameters: Sequence[FitParameter], scaled_values) -> np.ndarray:
    """The model at the given scaled values, called with physical values by name."""
    return np.asarray(model(x, **_values_dict(parameters, scaled_values)))


def _residual_vector(model, x, data, scales, parameters, scaled_values) -> np.ndarray:
    """What the optimiser minimises the squares of: weighted normalized data residuals, then prior residuals.

    Non-finite model values become a large finite residual, so the optimiser
    steps back from a region where the model breaks down instead of stopping.
    """
    # Data part: compare, stack Re and Im, times weight / sigma.
    _, curve = _matching_data_and_model(data, _evaluate_model(model, x, parameters, scaled_values))
    residuals = _stack_real_imag(data - curve) * scales.fit_factor
    residuals = np.where(np.isfinite(residuals), residuals, _LARGE_RESIDUAL)

    # Prior part: one extra residual (value - mean) / std per parameter with a prior, in the scaled space.
    prior_residuals = []
    for parameter, value in zip(parameters, scaled_values):
        prior = _scaled_prior(parameter)
        if prior is not None:
            prior_residuals.append((value - prior[0]) / prior[1])
    return np.concatenate([residuals, prior_residuals]) if prior_residuals else residuals


def _relative_diff_steps(parameters, start_physical, start_scaled, jac) -> np.ndarray | None:
    """scipy's ``diff_step`` array from the parameters' absolute steps, or None if no parameter sets one.

    scipy takes a *relative* step: the actual step is ``diff_step * max(1, |x|)``
    (x in the scaled space). The absolute step wanted, converted to the scaled
    space at the start value, is divided by that factor; parameters without a
    ``diff_step`` keep scipy's default relative step for the chosen scheme.
    """
    # Nothing to do when every parameter keeps the default.
    if all(parameter.diff_step is None for parameter in parameters):
        return None

    # scipy's defaults: square root (2-point) or cube root (3-point) of the machine precision.
    default = np.finfo(float).eps ** (1.0 / 3.0 if jac == "3-point" else 0.5)
    steps = []
    for parameter, scaled in zip(parameters, start_scaled):
        absolute = _scaled_diff_step(parameter, start_physical[parameter.name])
        steps.append(default if absolute is None else absolute / max(1.0, abs(scaled)))
    return np.array(steps)


def _covariance_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """``(JᵀJ)⁻¹`` via the singular values of J; all inf (with a warning) when J does not have full rank.

    A singular value that is zero to rounding means a direction in parameter
    space the residuals do not change along: the data cannot determine it,
    and a pseudo-inverse would report a misleading zero uncertainty there.
    """
    # Singular values below the rounding threshold mark undetermined directions.
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    threshold = np.finfo(float).eps * max(jacobian.shape) * (singular_values[0] if singular_values.size else 0.0)
    n_parameters = jacobian.shape[1]
    if singular_values.size < n_parameters or not singular_values[0] > 0 or np.any(singular_values <= threshold):
        warnings.warn(
            "The covariance could not be estimated: the data do not determine every parameter "
            "(the Jacobian does not have full rank). Fix a parameter, add a prior or add data.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.full((n_parameters, n_parameters), np.inf)
    return (right_vectors.T / singular_values**2) @ right_vectors
