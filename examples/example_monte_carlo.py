"""The two Monte Carlo studies on the same data: two peaks, a big one at X = 5 and a small one at X = -5.

1. monte_carlo_convergence_fit -- *where does a fit end from a poor start?* One Gaussian is fitted 200
   times to the SAME data, each time from a different start position (0 +- 5). Each run ends on one of the
   two peak maxima: the big peak is the best fit (lowest cost), the small one a local minimum. The end
   points map where the maxima are and which starts lead to which; their spread is not an uncertainty --
   the data never change.
2. monte_carlo_noise_fit -- *how precise is the fit?* Two Gaussians are fitted 5000 times, each time to a
   fresh pseudo-measurement (the true curve plus new noise). The histogram of the fitted position of the
   big peak is the real spread of the fit; the terminal output compares it with the uncertainty each fit
   reports from its covariance (the median over the runs). Where they agree, a single fit can be trusted.

Figures: the data (``_data.png``), the start and end points of the convergence study (``_end_points.png``)
and the fitted peak positions of the noise study (``_noise.png``).
"""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, plot_histogram_1d, style_colors

from cjcfit import FitParameter, add_noise, monte_carlo_convergence_fit, monte_carlo_noise_fit, noise_sigma

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]


# The models at module level, so the worker processes can import them (a lambda or a nested function cannot
# be sent to them).
def gaussian(x, amplitude, center, width):
    """One Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def two_peaks(x, amplitude_1, center_1, width_1, amplitude_2, center_2, width_2):
    """Two Gaussian peaks."""
    return gaussian(x, amplitude_1, center_1, width_1) + gaussian(x, amplitude_2, center_2, width_2)


# The guard is needed for n_processes > 1: each worker process imports this script again (Windows, macOS).
if __name__ == "__main__":
    # The data: big peak at +5, small one at -5, 81 points with noise 0.03.
    TRUE = {"amplitude_1": 0.8, "center_1": 5.0, "width_1": 1.0, "amplitude_2": 0.4, "center_2": -5.0, "width_2": 1.0}
    x = np.linspace(-10.0, 10.0, 81)
    sigma = 0.03
    data = add_noise(two_peaks(x, **TRUE), additive=sigma, seed=9)

    # Figure 1: the data and the true curve with its two maxima.
    x_fine = np.linspace(-10.0, 10.0, 800)
    plot_1d(
        [x, x_fine], [data, two_peaks(x_fine, **TRUE)],
        yerr_data=[np.full(x.size, sigma), None],
        colors=[BLUE, RED], err_colors=[BLUE],
        line_show=[False, True], scatter_show=[True, False], line_styles=["-", "-"],
        legend_labels=["data", "true"],
        xlabel="X axis [arb. u]", ylabel="Y axis [arb. u]",
        xlim=(-10.0, 10.0), ylim=(-0.1, 1.0),
        legend_loc="upper left",
        fig_aspect_ratio=(0.5, 0.5),
        save_path=f"{output_stem}_data.png",
        fig_show=False,
    )

    # Study 1, convergence: one Gaussian from 200 starts, centre anywhere around 0 (spread 5).
    single_peak = [
        FitParameter("amplitude", 0.5, bounds=(0.0, None)),
        FitParameter("center", 0.0, initial_std=5.0),
        FitParameter("width", 1.5, bounds=(0.1, 10.0)),
    ]
    convergence = monte_carlo_convergence_fit(
        gaussian, x, data, single_peak, sigma=sigma, n_runs=200, seed=9, n_processes=4,
    )
    ends = convergence.values["center"]
    on_big, on_small = np.abs(ends - 5.0) < 0.5, np.abs(ends + 5.0) < 0.5
    print(f"convergence: big peak {on_big.sum()} runs, small peak {on_small.sum()} runs, elsewhere {(~on_big & ~on_small).sum()}")
    print(f"best fit: {convergence.best_values}, cost {convergence.best_cost:.1f}")

    # Figure 2: where the runs started, split by where they ended (dark grey: the global minimum at the big
    # peak, light grey: elsewhere, i.e. the local minimum) and stacked, with the end points (blue) on top.
    # Each minimum collects the starts from one connected range of X (its basin), so a bin only holds both
    # colours where it straddles the border between two basins.
    starts = convergence.start_values["center"]
    bins = np.linspace(-10.0, 10.0, 41)
    _, ax = plot_histogram_1d(
        [starts[on_big], starts[~on_big]], bins=bins, histtype="barstacked",
        colors=["dimgray", "silver"], alpha=0.9,
        legend_labels=["start → global", "start → local"],
        legend_show=False,
        fig_aspect_ratio=(0.5, 0.5),
        fig_show=False,
    )
    plot_histogram_1d(
        ends, bins=bins,
        colors=[BLUE], alpha=0.8,
        legend_labels=["end"],
        xlabel="X axis [arb. u]", ylabel="Runs",
        xlim=(-10.0, 10.0), ylim=(0.0, 200.0),
        legend_loc="upper left",
        ax=ax,
        save_path=f"{output_stem}_end_points.png",
    )

    # Study 2, noise: two Gaussians fitted to 5000 fresh pseudo-measurements, from starts 10-20 % off,
    # weighted with the noise's own sigma.
    two_peak_parameters = [
        FitParameter("amplitude_1", 0.7, bounds=(0.0, None)),
        FitParameter("center_1", 4.7),
        FitParameter("width_1", 1.2, bounds=(0.1, 10.0)),
        FitParameter("amplitude_2", 0.3, bounds=(0.0, None)),
        FitParameter("center_2", -4.7),
        FitParameter("width_2", 1.2, bounds=(0.1, 10.0)),
    ]
    noise = monte_carlo_noise_fit(
        two_peaks, x, TRUE, two_peak_parameters,
        additive=sigma, sigma=noise_sigma(two_peaks(x, **TRUE), x, additive=sigma),
        n_runs=5000, seed=8, n_processes=4, show_progress=True,
    )
    for name in noise.parameter_names:
        print(
            f"{name:11s} bias {noise.bias[name]:+.4f}  spread {noise.std[name]:.4f}  "
            f"reported {noise.median_uncertainty[name]:.4f}"
        )
    print(f"failed fits: {noise.n_failed}, at a bound: {noise.n_at_bound}")

    # Figure 3: the fitted position of the big peak, in the style of figure 2.
    plot_histogram_1d(
        noise.values["center_1"], bins=25, bin_range=(4.9, 5.1),
        colors=[BLUE], alpha=0.8,
        legend_labels=["5000 MC runs"],
        xlabel="X axis [arb. u]", ylabel="Count [arb. u]",
        xlim=(4.9, 5.1), ylim=(0.0, 1000.0),
        legend_loc="upper left",
        fig_aspect_ratio=(0.5, 0.5),
        save_path=f"{output_stem}_noise.png",
        fig_show=False,
    )
