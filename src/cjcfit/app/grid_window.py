"""The grid window: chi-squared map over two parameters, click a start point, close to accept.

Layout: the map on the left with a red + at the selected point, the data
(solid blue) and the model at that point (dashed yellow) on the right, Re and
Im for complex data, and the selected values in the title. The map shows
chi-squared on a log colour scale with the colormap reversed, so the best
(lowest) region is bright.
"""

from __future__ import annotations

from typing import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from cjcfit import discrete_grid_fit

DATA_COLOR, MODEL_COLOR = "#003399", "#FFCC00"


def pick_initial_values(
    model: Callable,
    x,
    data,
    grid: Mapping[str, object],
    *,
    sigma=None,
    weights=None,
    x_label: str = "x",
) -> dict[str, float]:
    """Show the chi-squared map over two grid parameters and return the start values clicked.

    Computes :func:`cjcfit.discrete_grid_fit` (one model evaluation per grid
    point), opens the window and waits until it is closed. Click on the map
    to select a grid point (the nearest one; on log-spaced axes nearest in
    the logarithm); the model curve at that point is drawn over the data.

    Parameters
    ----------
    model, x, data, sigma, weights
        As for :func:`cjcfit.discrete_grid_fit`.
    grid : mapping of name -> 1D array-like
        Exactly two parameters: the first on the horizontal axis, the second
        on the vertical one. An axis whose values are evenly spaced in the
        logarithm is drawn on a log scale.
    x_label : str, default "x"
        Label of the horizontal axis of the data panels.

    Returns
    -------
    dict[str, float]
        The selected values (the best grid point if nothing was clicked),
        ready for ``initial_values`` of :func:`cjcfit.fit_least_squares`.
    """
    # Two parameters, because the window is a 2D map.
    if len(grid) != 2:
        raise ValueError(f"pick_initial_values needs exactly two grid parameters, got {len(grid)}.")
    result = discrete_grid_fit(model, x, data, grid, sigma=sigma, weights=weights)
    (x_name, y_name), data = result.parameter_names, np.asarray(data)
    x_values, y_values = result.grid[x_name], result.grid[y_name]
    x_is_log, y_is_log = _is_log_spaced(x_values), _is_log_spaced(y_values)
    selected = list(result.best_index)

    # Figure: map on the left, one or two data panels on the right.
    fig = plt.figure(layout="constrained")
    fig.set_size_inches(12.0, 7.0)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1])
    ax_map = fig.add_subplot(outer[0])
    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.35)
    ax_top, ax_bottom = fig.add_subplot(right[0]), fig.add_subplot(right[1])

    # Chi-squared map (pcolormesh wants y rows, x columns: the transpose), log colours; inf cells stay blank.
    finite = result.chi_squared[np.isfinite(result.chi_squared)]
    positive = finite[finite > 0]
    lowest = positive.min() if positive.size else 1e-300
    shown = np.where(np.isfinite(result.chi_squared), np.maximum(result.chi_squared, lowest), np.nan).T
    mesh = ax_map.pcolormesh(
        _cell_edges(x_values, x_is_log), _cell_edges(y_values, y_is_log), shown,
        cmap="inferno_r", norm="log", vmin=lowest, vmax=max(lowest * 10, np.nanmax(shown)),
    )
    fig.colorbar(mesh, ax=ax_map, label="χ²", shrink=0.9)
    ax_map.set_xscale("log" if x_is_log else "linear")
    ax_map.set_yscale("log" if y_is_log else "linear")
    ax_map.set_xlabel(x_name)
    ax_map.set_ylabel(y_name)
    ax_map.set_title("Click to select — close when done")
    (marker,) = ax_map.plot([x_values[selected[0]]], [y_values[selected[1]]], "r+", ms=14, mew=2, zorder=5)

    # Data panels: Re and Im for complex data; data and residuals (data - model) for real data.
    # The data are drawn against x when x is a matching 1D axis, else against the point index.
    is_complex = np.iscomplexobj(data)
    x_axis = np.asarray(x) if np.ndim(x) == 1 and np.size(x) == data.size else np.arange(data.size)
    if is_complex:
        parts = [(ax_top, np.real, "Re"), (ax_bottom, np.imag, "Im")]
    else:
        parts = [(ax_top, np.real, "data"), (ax_bottom, None, "residual")]
    model_lines = []
    for ax, part, label in parts:
        if part is not None:
            ax.plot(x_axis, part(data).ravel(), "-", color=DATA_COLOR, lw=1.5)
        model_lines.append(ax.plot(x_axis, np.zeros(data.size), "--", color=MODEL_COLOR, lw=1.5)[0])
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)

    def show_selection() -> None:
        """Move the marker, redraw the model at the selected point and put its values in the title."""
        values = _selected_values(result, selected)
        curve = np.asarray(model(x, **values)).ravel()
        marker.set_data([values[x_name]], [values[y_name]])
        if is_complex:
            model_lines[0].set_ydata(curve.real)
            model_lines[1].set_ydata(curve.imag)
        else:
            model_lines[0].set_ydata(np.real(curve))
            model_lines[1].set_ydata(data.ravel() - np.real(curve))
        for ax in (ax_top, ax_bottom):
            ax.relim()
            ax.autoscale_view()
        chi_squared = result.chi_squared[tuple(selected)]
        fig.suptitle(f"{x_name} = {values[x_name]:.4g}    {y_name} = {values[y_name]:.4g}    χ² = {chi_squared:.4g}", fontsize=11)
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        """Select the grid point nearest a left click on the map."""
        if event.inaxes is not ax_map or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        selected[0] = _nearest_index(x_values, event.xdata, x_is_log)
        selected[1] = _nearest_index(y_values, event.ydata, y_is_log)
        show_selection()

    # Show, wait for the window to close, return what was selected last.
    show_selection()
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    return _selected_values(result, selected)


def _selected_values(result, selected: list[int]) -> dict[str, float]:
    """The parameter values at grid index ``selected``."""
    return {name: float(result.grid[name][i]) for name, i in zip(result.parameter_names, selected)}


def _is_log_spaced(values: np.ndarray) -> bool:
    """True for at least three positive values that are evenly spaced in the logarithm but not linearly."""
    if values.size < 3 or np.any(values <= 0):
        return False
    log_steps, steps = np.diff(np.log(values)), np.diff(values)
    return bool(np.allclose(log_steps, log_steps[0], rtol=1e-6) and not np.allclose(steps, steps[0], rtol=1e-6))


def _cell_edges(values: np.ndarray, log: bool) -> np.ndarray:
    """Cell edges halfway between the centres (in the logarithm for log axes), extended at both ends."""
    # One value has no neighbour: give it a cell of +-50 % (or +-1 for zero).
    centres = np.log10(values) if log else values.astype(np.float64)
    if centres.size == 1:
        half = 0.5 * (abs(centres[0]) or 1.0) if not log else 0.5
        edges = np.array([centres[0] - half, centres[0] + half])
    else:
        middle = 0.5 * (centres[:-1] + centres[1:])
        edges = np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])
    return 10.0**edges if log else edges


def _nearest_index(values: np.ndarray, clicked: float, log: bool) -> int:
    """Index of the grid value nearest the clicked one (nearest in the logarithm on log axes)."""
    if log:
        return int(np.argmin(np.abs(np.log10(values) - np.log10(max(clicked, 1e-300)))))
    return int(np.argmin(np.abs(values - clicked)))
