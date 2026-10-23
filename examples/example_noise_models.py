"""The noise model's three forms: a number, an array (one sigma per point) and a function of x."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors

from cjcfit import add_noise, noise_sigma

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]

# A noise-free curve on X = 0..1, between -1 and 1 and centred on 0.
x = np.linspace(0.0, 1.0, 61)
signal = -0.8 * np.sin(2 * np.pi * x)

# Three additive noise terms of similar size:
#   number:   the same sigma everywhere;
#   array:    one sigma per point, here from a table (e.g. a measured noise floor), a step at X = 0.5;
#   function: sigma as a formula in x, evaluated by add_noise at the x it is given (a bathtub here,
#             large at both ends of the range, as the noise of a spectrum is outside its usable band).
number = 0.2
array = np.where(x < 0.5, 0.1, 0.3)


def bathtub(x):
    """Noise that rises steeply at both ends of the range."""
    return 0.08 + 0.8 * np.exp(-x / 0.08) + 0.8 * np.exp((x - 1.0) / 0.08)


# The sigma of each form, from noise_sigma (also what a fit of these data should get as sigma).
forms = {"number": number, "array": array, "function": bathtub}
sigmas = [noise_sigma(signal, x, additive=term) for term in forms.values()]
plot_1d(
    [x, x, x], sigmas,
    colors=[BLUE, YELLOW, RED], line_styles=["-", "--", "-"],
    legend_labels=["number: 0.2", "array: a step", "function: bathtub(x)"],
    xlabel="X axis [arb. u]", ylabel=r"$\sigma$ [arb. u]",
    xlim=(0.0, 1.0), ylim=(0.0, 1.0),
    legend_loc="upper center",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_sigma.png",
    fig_show=False,
)

# One noisy realisation per form, each in its own figure, with its sigma as error bars; the legend names
# the form, so each figure can be read on its own.
data_labels = {"number": r"$\sigma$ = 0.2", "array": r"$\sigma$: step", "function": r"$\sigma$: bathtub"}
for (name, term), sigma in zip(forms.items(), sigmas):
    noisy = add_noise(signal, x, additive=term, seed=5)
    plot_1d(
        [x, x], [noisy, signal],
        yerr_data=[sigma, None],
        colors=[BLUE, RED], err_colors=[BLUE],
        line_show=[False, True], scatter_show=[True, False], line_styles=["-", "-"],
        legend_labels=[data_labels[name], "true"],
        xlabel="X axis [arb. u]", ylabel="Y axis [arb. u]",
        xlim=(0.0, 1.0), ylim=(-1.0, 1.0),
        legend_loc="lower right",
        fig_aspect_ratio=(0.5, 0.5),
        save_path=f"{output_stem}_{name}.png",
        fig_show=False,
    )
