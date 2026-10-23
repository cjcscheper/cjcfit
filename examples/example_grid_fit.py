"""Discrete grid fit: an R² map over two parameters shows where the fit can start, and the best point.

The model is a sine, 0.8 sin(2 pi (2 x1 x + x2)): x1 sets the number of periods on X = 0..1 (one at x1 = 0.5),
x2 the phase in periods. Its R² over (x1, x2) is a single tilted ridge through the true values (a little
more frequency can be traded for a little less phase) and about 0 everywhere else, where the model does not
resemble the data at all. A least-squares fit started out there can stop on the edge of the allowed range;
the map shows the good region at a glance, and its best point is a safe start.
"""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, plot_2d, style_colors

from cjcfit import FitParameter, add_noise, discrete_grid_fit, fit_least_squares

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]


def sine(x, x1, x2):
    """0.8 sin(2 pi (2 x1 x + x2)): amplitude 0.8, 2 x1 periods on X = 0..1, phase x2 periods."""
    return 0.8 * np.sin(2 * np.pi * (2 * x1 * x + x2))


# Data: the sine at x1 = x2 = 0.5 (one period, half a period of phase) at 26 points, noise 0.05.
TRUE = {"x1": 0.5, "x2": 0.5}
x = np.linspace(0.0, 1.0, 26)
sigma = 0.05
data = add_noise(sine(x, **TRUE), additive=sigma, seed=1)

# The grid over x1 and x2 from 0 to 1. Without sigma, chi-squared is the plain sum of squared residuals,
# and R^2 = 1 - chi^2 / sum((data - mean)^2) follows it exactly.
grid = {"x1": np.linspace(0.0, 1.0, 101), "x2": np.linspace(0.0, 1.0, 101)}
result = discrete_grid_fit(sine, x, data, grid)
r_squared = 1.0 - result.chi_squared / np.sum((data - data.mean()) ** 2)
print(f"best grid point: {result.best_values}, R^2 = {r_squared[result.best_index]:.3f}")

# A least-squares fit started far from the ridge, at (0.95, 0.5), runs into the bound x1 = 1 and stops there.
parameters = [FitParameter("x1", 0.95, bounds=(0.0, 1.0)), FitParameter("x2", 0.5)]
stuck = fit_least_squares(sine, x, data, parameters, sigma=sigma)
print(f"fit started at (0.95, 0.5): x1 = {stuck.values['x1']:.3f}, at a bound: {stuck.at_bound}, R^2 = {stuck.r_squared:.2f}")

# The R^2 map (axis 0 of the grid is x1, plot_2d wants rows = y: the transpose), clipped to 0..1.
plot_2d(
    grid["x1"], grid["x2"], np.clip(r_squared, 0.0, 1.0).T,
    xlabel="X1 axis [arb. u]", ylabel="X2 axis [arb. u]", cbar_label=r"$R^2$",
    clim=(0.0, 1.0), xlim=(0.0, 1.0), ylim=(0.0, 1.0),
    fig_aspect_ratio=(0.6, 0.5),
    save_path=f"{output_stem}_map.png",
    fig_show=False,
)

# The least-squares fit from the best grid point.
fit = fit_least_squares(sine, x, data, parameters, sigma=sigma, initial_values=result.best_values)
for name in fit.parameter_names:
    print(f"{name} = {fit.values[name]:.4f} +- {fit.uncertainties[name]:.4f}   (true {TRUE[name]})")

# Data, fit and true curve.
x_fine = np.linspace(0.0, 1.0, 400)
plot_1d(
    [x, x_fine, x_fine],
    [data, sine(x_fine, **fit.values), sine(x_fine, **TRUE)],
    yerr_data=[np.full(x.size, sigma), None, None],
    colors=[BLUE, YELLOW, RED], err_colors=[BLUE],
    line_show=[False, True, True], scatter_show=[True, False, False], line_styles=["-", "--", "-"],
    legend_labels=["data", "fit", "true"],
    xlabel="X axis [arb. u]", ylabel="Y axis [arb. u]",
    xlim=(0.0, 1.0), ylim=(-1.0, 1.0),
    legend_loc="upper left",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_data.png",
    fig_show=False,
)
