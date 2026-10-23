"""Fit parameters: what is fitted, where it may go, and in which space the optimiser moves it.

A :class:`FitParameter` holds everything about one fitted quantity in physical
units: its start value, bounds, optional Gaussian prior and finite-difference
step. The optimiser does not always work in those units: with ``scale`` it
moves ``log10(value)``, ``ln(value)`` or ``value / normalization_factor``
instead (THEORY.md, section 5, says why). :func:`to_scaled` and
:func:`to_physical` convert between the two; everything else here is the
bookkeeping that follows from them (bounds, priors, steps and spreads in the
scaled space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Scale = Literal["log10", "ln", "normalized"]

_SCALES = (None, "log10", "ln", "normalized")


@dataclass(frozen=True)
class FitParameter:
    """One fitted quantity, everything in physical units.

    Parameters
    ----------
    name : str
        Name of the parameter; the model function must accept a keyword
        argument with exactly this name, and results are keyed by it.
    initial : float
        Start value of the fit. Must lie inside ``bounds``.
    bounds : (lower, upper), optional
        Physical limits; either side may be None for "no limit". The fit never
        leaves them. For log scales both given limits must be positive.
    scale : {None, "log10", "ln", "normalized"}, optional
        Space in which the optimiser moves the parameter: None moves the value
        itself, "log10"/"ln" its logarithm (for positive quantities that may
        span decades, so a step is a fixed *factor*), "normalized" the value
        divided by ``normalization_factor`` (to bring e.g. a thickness of 1e-6 m
        to order 1, which the optimiser's tolerances assume).
    normalization_factor : float, optional
        Positive divisor for ``scale="normalized"``; required there, not
        allowed with any other scale (a forgotten ``scale`` would otherwise go
        unnoticed).
    prior_mean, prior_std : float, optional
        Gaussian prior (prior knowledge from elsewhere, e.g. a thickness
        measured with a micrometer): adds ``(value - prior_mean) / prior_std``
        as an extra residual. Give both or neither; ``prior_std > 0``. For log
        scales the prior is Gaussian in the logarithm, with the spread
        converted to first order (THEORY.md, section 6).
    diff_step : float, optional
        Absolute finite-difference step for the Jacobian, in physical units.
        Needed when the model has numerical noise of its own (an ODE or PDE
        solver, an interpolated table), where the default step (about 1e-8
        relative) only differentiates that noise. :func:`suggest_diff_step`
        gives a starting value.
    initial_std : float, optional
        Spread of the start values in :func:`cjcfit.monte_carlo_convergence_fit`
        and in :func:`cjcfit.monte_carlo_noise_fit` with ``random_start=True``,
        in physical units: each run starts at ``initial ± initial_std`` (drawn
        normally; for log scales log-normally with the same spread to first
        order, so values stay positive). None or 0: always start at ``initial``.
        Ignored by every other function.

    Raises
    ------
    ValueError
        For inconsistent settings, with a message naming the parameter.
    """

    name: str
    initial: float
    bounds: tuple[float | None, float | None] | None = None
    scale: Scale | None = None
    normalization_factor: float | None = None
    prior_mean: float | None = None
    prior_std: float | None = None
    diff_step: float | None = None
    initial_std: float | None = None

    def __post_init__(self) -> None:
        # The name becomes a keyword argument of the model, so it must be a valid identifier.
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(
                f"FitParameter name {self.name!r} must be a valid Python identifier, "
                "because the model function receives it as a keyword argument."
            )
        label = f"FitParameter {self.name!r}"
        if not math.isfinite(self.initial):
            raise ValueError(f"{label}: initial must be finite, got {self.initial}.")

        # The scale and its normalization factor belong together.
        if self.scale not in _SCALES:
            raise ValueError(f"{label}: scale must be None, 'log10', 'ln' or 'normalized', got {self.scale!r}.")
        if self.scale == "normalized":
            if self.normalization_factor is None or not self.normalization_factor > 0:
                raise ValueError(f"{label}: scale='normalized' needs a positive normalization_factor.")
        elif self.normalization_factor is not None:
            raise ValueError(f"{label}: normalization_factor is only used with scale='normalized'.")
        is_log = self.scale in ("log10", "ln")
        if is_log and self.initial <= 0:
            raise ValueError(f"{label}: a log scale needs a positive initial value, got {self.initial}.")

        # Bounds: ordered, positive for log scales, and the start value inside them.
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError(f"{label}: bounds must be (lower, upper), either side may be None.")
            lower, upper = self.bounds
            if lower is not None and upper is not None and not lower < upper:
                raise ValueError(f"{label}: bounds lower ({lower}) must be below upper ({upper}).")
            if is_log and any(bound is not None and bound <= 0 for bound in (lower, upper)):
                raise ValueError(f"{label}: bounds must be positive for a log scale, got {self.bounds}.")
            if (lower is not None and self.initial < lower) or (upper is not None and self.initial > upper):
                raise ValueError(f"{label}: initial ({self.initial}) lies outside bounds {self.bounds}.")

        # A prior needs both numbers, a positive spread and, for log scales, a positive mean.
        if (self.prior_mean is None) != (self.prior_std is None):
            raise ValueError(f"{label}: give prior_mean and prior_std together, or neither.")
        if self.prior_std is not None and not self.prior_std > 0:
            raise ValueError(f"{label}: prior_std must be positive, got {self.prior_std}.")
        if is_log and self.prior_mean is not None and self.prior_mean <= 0:
            raise ValueError(f"{label}: a log scale needs a positive prior_mean, got {self.prior_mean}.")

        # Steps and spreads are sizes: positive and non-negative.
        if self.diff_step is not None and not self.diff_step > 0:
            raise ValueError(f"{label}: diff_step must be positive, got {self.diff_step}.")
        if self.initial_std is not None and not self.initial_std >= 0:
            raise ValueError(f"{label}: initial_std must be zero or positive, got {self.initial_std}.")


def to_scaled(parameter: FitParameter, value):
    """Physical value(s) of ``parameter`` -> the optimiser's scaled space.

    ``log10(value)``, ``ln(value)``, ``value / normalization_factor`` or the
    value itself, depending on ``parameter.scale``. Works on numbers and arrays.
    """
    # One line per scale; log scales reject values that have no logarithm.
    value = np.asarray(value, dtype=np.float64)
    if parameter.scale in ("log10", "ln") and np.any(value <= 0):
        raise ValueError(f"FitParameter {parameter.name!r}: a log scale needs positive values, got {value}.")
    if parameter.scale == "log10":
        scaled = np.log10(value)
    elif parameter.scale == "ln":
        scaled = np.log(value)
    elif parameter.scale == "normalized":
        scaled = value / parameter.normalization_factor
    else:
        scaled = value
    return float(scaled) if scaled.ndim == 0 else scaled


def to_physical(parameter: FitParameter, scaled_value):
    """Scaled value(s) of ``parameter`` (the optimiser's space) -> physical units; inverse of :func:`to_scaled`."""
    # One line per scale, the inverse of to_scaled.
    scaled_value = np.asarray(scaled_value, dtype=np.float64)
    if parameter.scale == "log10":
        value = 10.0**scaled_value
    elif parameter.scale == "ln":
        value = np.exp(scaled_value)
    elif parameter.scale == "normalized":
        value = scaled_value * parameter.normalization_factor
    else:
        value = scaled_value
    return float(value) if value.ndim == 0 else value


def suggest_diff_step(value: float, relative_step: float = 1e-3, absolute_floor: float | None = None) -> float:
    """A finite-difference step for :attr:`FitParameter.diff_step`: ``|value| * relative_step``.

    A step of 0.1 % of the value is large enough to rise above the numerical
    noise of a typical ODE/PDE solver or interpolated table, and small enough
    that the model is still close to linear over it. For ``value = 0`` the
    scale 1 is used; ``absolute_floor`` sets a minimum step (e.g. for a
    parameter that may pass through zero).
    """
    # Validate the two sizes before using them.
    if not relative_step > 0:
        raise ValueError(f"relative_step must be positive, got {relative_step}.")
    if absolute_floor is not None and not absolute_floor > 0:
        raise ValueError(f"absolute_floor must be positive, got {absolute_floor}.")

    # Relative step, with unit scale at zero and the optional floor.
    step = (abs(value) if value != 0 else 1.0) * relative_step
    return float(step if absolute_floor is None else max(step, absolute_floor))


def _scaled_bounds(parameter: FitParameter) -> tuple[float, float]:
    """Bounds in the scaled space; a missing side becomes -inf / +inf (for log scales: 0 -> -inf)."""
    # No bounds at all is the same as two open sides.
    lower, upper = parameter.bounds if parameter.bounds is not None else (None, None)
    return (
        -np.inf if lower is None else to_scaled(parameter, lower),
        np.inf if upper is None else to_scaled(parameter, upper),
    )


def _physical_per_scaled(parameter: FitParameter, value: float) -> float:
    """d(physical)/d(scaled) at ``value``: converts spreads and covariances between the two spaces."""
    # Derivative of to_physical at the point that corresponds to `value`.
    if parameter.scale == "log10":
        return math.log(10.0) * value
    if parameter.scale == "ln":
        return value
    if parameter.scale == "normalized":
        return parameter.normalization_factor
    return 1.0


def _scaled_std(parameter: FitParameter, std: float, value: float) -> float:
    """A physical spread ``std`` around ``value`` expressed in the scaled space, to first order."""
    return std / _physical_per_scaled(parameter, value)


def _scaled_prior(parameter: FitParameter) -> tuple[float, float] | None:
    """Prior mean and spread in the scaled space, or None without a prior (THEORY.md, section 6)."""
    # The mean is transformed exactly, the spread to first order around it.
    if parameter.prior_mean is None:
        return None
    return (
        to_scaled(parameter, parameter.prior_mean),
        _scaled_std(parameter, parameter.prior_std, parameter.prior_mean),
    )


def _scaled_diff_step(parameter: FitParameter, value: float) -> float | None:
    """The absolute physical ``diff_step`` as an absolute step in the scaled space, at ``value``."""
    # Exact difference of the scaled values one step apart; None leaves scipy's default.
    if parameter.diff_step is None:
        return None
    return to_scaled(parameter, value + parameter.diff_step) - to_scaled(parameter, value)
