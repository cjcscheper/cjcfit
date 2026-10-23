"""Chi-squared on a grid of parameter values: a search for a good starting point.

A least-squares fit only walks downhill from where it starts. When the cost
has several valleys, :func:`discrete_grid_fit` evaluates the model at every
combination of the given values and returns the cost map and its lowest point,
which is then a safe start for :func:`cjcfit.fit_least_squares` (``initial_values=result.best_values``).
THEORY.md, section 10. :func:`cjcfit.app.pick_initial_values` shows the same
map and lets you click the start point.

Without sigma and weights (or with one sigma for all points) the lowest
chi-squared is also the highest R², because ``R² = 1 - chi² / sum((data - mean)²)``
and the denominator depends on the data only. With sigma or weights that
differ per point, the map ranks by the weighted sum the fit minimises, so its
best point can differ from the one with the highest (unweighted) R²; the
map's point is the right start for that fit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from cjcfit.quality import _matching_data_and_model, _residual_scales, _stack_real_imag


@dataclass(frozen=True)
class GridFitResult:
    """Result of :func:`discrete_grid_fit`.

    Attributes
    ----------
    parameter_names : tuple[str, ...]
        Names in the order of ``grid``; axis k of ``chi_squared`` belongs to name k.
    grid : dict[str, ndarray]
        The values tried per parameter.
    chi_squared : ndarray
        Chi-squared at every grid point (with weights: the weighted sum the fit minimises), shape ``(len(grid[name_1]), len(grid[name_2]), ...)``:
        ``chi_squared[i, j]`` belongs to ``grid[name_1][i]``, ``grid[name_2][j]``.
        ``inf`` where the model returned NaN or inf. (For :func:`cjcplot.plot_2d`,
        which wants ``(len(y), len(x))``, pass ``chi_squared.T`` with name 1 on x.)
    best_index : tuple[int, ...]
        Index of the lowest chi-squared.
    best_values : dict[str, float]
        Parameter values at that point, ready for ``initial_values`` of a fit.
    best_chi_squared : float
    """

    parameter_names: tuple[str, ...]
    grid: dict[str, np.ndarray]
    chi_squared: np.ndarray
    best_index: tuple[int, ...]
    best_values: dict[str, float]
    best_chi_squared: float


def discrete_grid_fit(
    model: Callable, x, data, grid: Mapping[str, object], *, sigma=None, weights=None
) -> GridFitResult:
    """Chi-squared of ``model(x, **values)`` against ``data`` at every combination of the ``grid`` values.

    Parameters
    ----------
    model : callable
        As for :func:`cjcfit.fit_least_squares`; it receives the grid's names
        as keyword arguments (fix any others with ``functools.partial``).
    x : anything
        Passed to the model unchanged.
    data : array-like
        Real or complex measurement.
    grid : mapping of name -> 1D array-like
        Values to try per parameter, e.g. ``{"thickness": np.linspace(...),
        "density": np.logspace(...)}``. The number of model evaluations is the
        product of the lengths.
    sigma, weights : float or array-like, optional
        As for :func:`cjcfit.fit_least_squares`. The map is the sum the fit
        minimises, ``sum((weight * residual / sigma)²)``, so its lowest point
        is the best start for that fit; without either, the plain sum of
        squared residuals.

    Returns
    -------
    GridFitResult
    """
    # Grid axes: names usable as keyword arguments, 1D finite values.
    if not grid:
        raise ValueError("grid must give at least one parameter.")
    names = tuple(grid)
    axes = {}
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Grid name {name!r} must be a valid Python identifier (it becomes a keyword argument).")
        values = np.asarray(grid[name], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"grid[{name!r}] must be a non-empty 1D array of finite values.")
        axes[name] = values

    # Data, sigma and weights checked once for all grid points.
    data = np.asarray(data)
    data, _ = _matching_data_and_model(data, data)
    scales = _residual_scales(data, sigma, weights)

    # Chi-squared at every combination; a model that breaks down somewhere gives inf there.
    chi_squared = np.empty(tuple(axes[name].size for name in names))
    for index in itertools.product(*(range(axes[name].size) for name in names)):
        values = {name: float(axes[name][i]) for name, i in zip(names, index)}
        _, curve = _matching_data_and_model(data, model(x, **values))
        residuals = _stack_real_imag(data - curve) * scales.fit_factor
        chi_squared[index] = float(np.sum(residuals**2)) if np.all(np.isfinite(residuals)) else np.inf

    # The lowest point; if the model failed everywhere there is none.
    if not np.any(np.isfinite(chi_squared)):
        raise ValueError("The model returned NaN or inf at every grid point.")
    best_index = tuple(int(i) for i in np.unravel_index(int(np.argmin(chi_squared)), chi_squared.shape))
    return GridFitResult(
        parameter_names=names,
        grid=axes,
        chi_squared=chi_squared,
        best_index=best_index,
        best_values={name: float(axes[name][i]) for name, i in zip(names, best_index)},
        best_chi_squared=float(chi_squared[best_index]),
    )
