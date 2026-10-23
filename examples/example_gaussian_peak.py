"""Gaussian peak: a non-linear fit from a start that is off in position and height, with bounds."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors

from cjcfit import FitParameter, add_noise, fit_least_squares

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]


def gaussian(x, amplitude, center, width):
    """A Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


# The Gaussian of cjcplot's histogram example (centre 0, width 0.12, height 0.7), measured at 41 points
# with noise 0.03.
TRUE = {"amplitude": 0.7, "center": 0.0, "width": 0.12}
x = np.linspace(-1.0, 1.0, 41)
sigma = 0.03
data = add_noise(gaussian(x, **TRUE), additive=sigma, seed=4)

# The start: lower and shifted to the left. The model is non-linear in centre and width, so the fit walks
# downhill from there; bounds keep the amplitude and the width positive.
parameters = [
    FitParameter("amplitude", 0.4, bounds=(0.0, None)),
    FitParameter("center", -0.25),
    FitParameter("width", 0.12, bounds=(0.01, 1.0)),
]
fit = fit_least_squares(gaussian, x, data, parameters, sigma=sigma)
for name in fit.parameter_names:
    print(f"{name:9s} = {fit.values[name]:+.4f} +- {fit.uncertainties[name]:.4f}   (true {TRUE[name]})")
print(f"reduced chi-squared = {fit.reduced_chi_squared:.2f}, R^2 = {fit.r_squared:.4f}")

# Data, the fit, the true curve and (lowest) the start curve.
x_fine = np.linspace(-1.0, 1.0, 400)
start = {parameter.name: parameter.initial for parameter in parameters}
plot_1d(
    [x, x_fine, x_fine, x_fine],
    [data, gaussian(x_fine, **fit.values), gaussian(x_fine, **TRUE), gaussian(x_fine, **start)],
    yerr_data=[np.full(x.size, sigma), None, None, None],
    colors=[BLUE, YELLOW, RED, "gray"], err_colors=[BLUE],
    line_show=[False, True, True, True], scatter_show=[True, False, False, False],
    line_styles=["-", "--", "-", ":"],
    legend_labels=["data", "fit", "true", "start"],
    xlabel="X axis [arb. u]", ylabel="Y axis [arb. u]",
    xlim=(-1.0, 1.0), ylim=(-0.1, 1.0),
    legend_loc="upper left",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=str(output_stem.with_suffix(".png")),
    fig_show=False,
)
