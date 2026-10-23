"""Complex data: amplitude and delay of Y(x) = A exp(-i 2 pi x tau), fitted in Re and Im at once."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors

from cjcfit import FitParameter, add_noise, fit_least_squares

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]


def model_function(x, amplitude, delay):
    """Complex Y(x) = A exp(-i 2 pi x tau): magnitude A everywhere, phase falling linearly by 2 pi tau per unit x."""
    return amplitude * np.exp(-2j * np.pi * x * delay)


# Complex data on X = 0..1 (Re and Im of Y between -1 and 1, centred on 0): A = 0.5, tau = 0.9, so the phase
# falls to -2 pi * 0.9 at X = 1; additive noise of 0.025 on Re and on Im. A fit of Re and Im recovers A and
# tau from additive noise without bias. (Large *phase* noise would not: it pulls the average complex value
# towards zero, by exp(-b^2 / 2), and the fitted amplitude with it; THEORY.md, section 8.)
TRUE = {"amplitude": 0.5, "delay": 0.9}
x = np.linspace(0.04, 1.0, 25)
sigma = 0.025
data = add_noise(model_function(x, **TRUE), additive=sigma, seed=4)

# The start must be within about a quarter period at the largest x (1 / (4 * 1) = 0.25 in tau): further
# away, the fit can lock onto a phase that is off by 2 pi there (a local minimum).
parameters = [FitParameter("amplitude", 0.3, bounds=(0.0, None)), FitParameter("delay", 0.8)]
fit = fit_least_squares(model_function, x, data, parameters, sigma=sigma)
for name in fit.parameter_names:
    print(f"{name:9s} = {fit.values[name]:.4f} +- {fit.uncertainties[name]:.4f}   (true {TRUE[name]})")
print(f"reduced chi-squared = {fit.reduced_chi_squared:.2f} ({fit.degrees_of_freedom} degrees of freedom: Re and Im count)")

# |Y|: data with error bars sigma (to first order), fit and true magnitude.
x_fine = np.linspace(0.0, 1.0, 400)
plot_1d(
    [x, x_fine, x_fine],
    [np.abs(data), np.abs(model_function(x_fine, **fit.values)), np.abs(model_function(x_fine, **TRUE))],
    yerr_data=[np.full(x.size, sigma), None, None],
    colors=[BLUE, YELLOW, RED], err_colors=[BLUE],
    line_show=[False, True, True], scatter_show=[True, False, False], line_styles=["-", "--", "-"],
    legend_labels=["data", "fit", "true"],
    xlabel="X axis [arb. u]", ylabel=r"$|Y|$ [arb. u]",
    xlim=(0.0, 1.0), ylim=(0.0, 1.0),
    legend_loc="upper right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_amplitude.png",
    fig_show=False,
)

# Phase of Y: falls linearly, -2 pi x tau; error bars sigma / |Y| (rad), to first order.
plot_1d(
    [x, x_fine, x_fine],
    [
        np.unwrap(np.angle(data)),
        np.unwrap(np.angle(model_function(x_fine, **fit.values))),
        np.unwrap(np.angle(model_function(x_fine, **TRUE))),
    ],
    yerr_data=[sigma / np.abs(data), None, None],
    colors=[BLUE, YELLOW, RED], err_colors=[BLUE],
    line_show=[False, True, True], scatter_show=[True, False, False], line_styles=["-", "--", "-"],
    legend_labels=["data", "fit", "true"],
    xlabel="X axis [arb. u]", ylabel=r"$\angle Y$ [rad]",
    xlim=(0.0, 1.0), ylim=(-2 * np.pi, 0.0),
    yticks_pos=[-2 * np.pi, -1.5 * np.pi, -np.pi, -0.5 * np.pi, 0.0],
    yticks_labels=[r"$-2\pi$", r"$-3\pi/2$", r"$-\pi$", r"$-\pi/2$", "0"],
    legend_loc="upper right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_phase.png",
    fig_show=False,
)
