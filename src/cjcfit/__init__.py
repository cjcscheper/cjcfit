"""cjcfit: general least-squares fitting, noise models and Monte Carlo studies.

You write the model function, ``model(x, name_1, name_2, ...)``; cjcfit fits
it (:func:`fit_least_squares`, with bounds, log scaling, priors and complex
data), says how good the fit is (:func:`fit_quality`), makes noisy
pseudo-measurements (:func:`add_noise`, :func:`noise_sigma`), studies the
fit's precision and robustness (:func:`monte_carlo_noise_fit`,
:func:`monte_carlo_convergence_fit`) and searches a grid for a starting point
(:func:`discrete_grid_fit`). There are no model functions in the package and
no physics: numpy and scipy are the only dependencies.

The interactive start-point picker, :func:`cjcfit.app.pick_initial_values`,
lives in :mod:`cjcfit.app` and needs the ``app`` extra.
"""

from cjcfit.fitting import FitResult, fit_least_squares
from cjcfit.grid import GridFitResult, discrete_grid_fit
from cjcfit.monte_carlo import MonteCarloResult, monte_carlo_convergence_fit, monte_carlo_noise_fit
from cjcfit.noise import add_noise, noise_sigma
from cjcfit.parameters import FitParameter, suggest_diff_step, to_physical, to_scaled
from cjcfit.quality import FitQuality, fit_quality

__version__ = "0.1.0"

__all__ = [
    "FitParameter",
    "FitQuality",
    "FitResult",
    "GridFitResult",
    "MonteCarloResult",
    "add_noise",
    "discrete_grid_fit",
    "fit_least_squares",
    "fit_quality",
    "monte_carlo_convergence_fit",
    "monte_carlo_noise_fit",
    "noise_sigma",
    "suggest_diff_step",
    "to_physical",
    "to_scaled",
]
