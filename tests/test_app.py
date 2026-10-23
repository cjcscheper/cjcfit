"""cjcfit.app, headless: the grid window built, a click simulated, the selected values returned.

Skipped when the app extra (matplotlib, cjcplot) is not installed, as in CI.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("cjcplot")
# Draw in memory; CI machines have no screen. Must run before pyplot is imported; plt.show() then returns at once.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402

from cjcfit.app import grid_window, pick_initial_values  # noqa: E402

X = np.linspace(-5.0, 5.0, 21)
GRID = {"slope": np.linspace(1.0, 3.0, 11), "intercept": np.logspace(-1, 1, 9)}


def line(x, slope, intercept):
    """Straight line."""
    return slope * x + intercept


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the figures each test made."""
    yield
    plt.close("all")


def click_on_map(slope: float, intercept: float):
    """A replacement for plt.show that left-clicks the map at (slope, intercept) and returns."""
    def show(*args, **kwargs):
        fig = plt.gcf()
        ax_map = fig.axes[0]
        fig.canvas.draw()
        x_px, y_px = ax_map.transData.transform((slope, intercept))
        fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, x_px, y_px, button=1))

    return show


def test_without_a_click_the_best_grid_point_is_returned(monkeypatch):
    """Closing the window at once returns the lowest chi-squared point (2.0, 1.0 on these grids)."""
    monkeypatch.setattr(grid_window.plt, "show", lambda *args, **kwargs: None)
    values = pick_initial_values(line, X, line(X, 2.0, 1.0), GRID, sigma=0.1)
    assert values == pytest.approx({"slope": 2.0, "intercept": 1.0})


def test_a_click_selects_the_nearest_grid_point(monkeypatch):
    """A click near (2.55, 3.0) selects slope 2.6 and, on the log axis, intercept 10^0.5 = 3.16."""
    monkeypatch.setattr(grid_window.plt, "show", click_on_map(2.55, 3.0))
    values = pick_initial_values(line, X, line(X, 2.0, 1.0) + 0j, GRID, x_label="x [mm]")
    assert values == pytest.approx({"slope": 2.6, "intercept": 10**0.5})
    # The log-spaced intercept axis is drawn on a log scale; complex data give Re and Im panels.
    fig = plt.gcf()
    assert fig.axes[0].get_yscale() == "log" and fig.axes[0].get_xscale() == "linear"
    assert [ax.get_ylabel() for ax in fig.axes[1:3]] == ["Re", "Im"]


def test_needs_exactly_two_parameters():
    """A grid of one parameter raises a ValueError."""
    with pytest.raises(ValueError, match="exactly two"):
        pick_initial_values(line, X, X, {"slope": [1.0, 2.0]})
