"""FitParameter validation, the scaled-space conversions and suggest_diff_step."""

import math

import numpy as np
import pytest

from cjcfit import FitParameter, suggest_diff_step, to_physical, to_scaled


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"name": "two words", "initial": 1.0}, "valid Python identifier"),
        ({"name": "a", "initial": math.nan}, "initial must be finite"),
        ({"name": "a", "initial": 1.0, "scale": "log2"}, "scale must be"),
        ({"name": "a", "initial": 1.0, "scale": "normalized"}, "positive normalization_factor"),
        ({"name": "a", "initial": 1.0, "normalization_factor": 2.0}, "only used with scale='normalized'"),
        ({"name": "a", "initial": -1.0, "scale": "log10"}, "positive initial"),
        ({"name": "a", "initial": 1.0, "bounds": (2.0, 0.0)}, "must be below upper"),
        ({"name": "a", "initial": 1.0, "bounds": (0.0, None), "scale": "ln"}, "positive for a log scale"),
        ({"name": "a", "initial": 5.0, "bounds": (0.0, 2.0)}, "outside bounds"),
        ({"name": "a", "initial": 1.0, "prior_mean": 1.0}, "prior_mean and prior_std together"),
        ({"name": "a", "initial": 1.0, "prior_mean": 1.0, "prior_std": 0.0}, "prior_std must be positive"),
        ({"name": "a", "initial": 1.0, "scale": "log10", "prior_mean": -1.0, "prior_std": 1.0}, "positive prior_mean"),
        ({"name": "a", "initial": 1.0, "diff_step": -1e-3}, "diff_step must be positive"),
        ({"name": "a", "initial": 1.0, "initial_std": -1.0}, "initial_std must be zero or positive"),
    ],
)
def test_invalid_settings_raise_clear_errors(settings, message):
    """Every inconsistent FitParameter raises a ValueError that says what is wrong."""
    with pytest.raises(ValueError, match=message):
        FitParameter(**settings)


def test_open_bounds_and_frozen():
    """Either side of the bounds may be None, and a FitParameter cannot be changed afterwards."""
    parameter = FitParameter("a", 1.0, bounds=(None, 2.0))
    with pytest.raises(AttributeError):
        parameter.initial = 3.0  # frozen dataclass


@pytest.mark.parametrize(
    ("scale", "factor", "physical", "scaled"),
    [(None, None, 3.0, 3.0), ("log10", None, 1e5, 5.0), ("ln", None, math.e**2, 2.0), ("normalized", 1e-6, 2e-6, 2.0)],
)
def test_scaled_space_round_trip(scale, factor, physical, scaled):
    """to_scaled gives log10, ln, value/factor or the value itself, and to_physical undoes it."""
    parameter = FitParameter("a", physical, scale=scale, normalization_factor=factor)
    assert to_scaled(parameter, physical) == pytest.approx(scaled)
    assert to_physical(parameter, scaled) == pytest.approx(physical)


def test_conversions_work_on_arrays():
    """Arrays are converted element by element; a log scale refuses non-positive values."""
    parameter = FitParameter("rate", 1.0, scale="log10")
    assert np.allclose(to_scaled(parameter, [1.0, 10.0, 100.0]), [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="positive values"):
        to_scaled(parameter, [1.0, 0.0])


def test_suggest_diff_step():
    """0.1 % of the value by default, unit scale at zero, and never below the floor."""
    assert suggest_diff_step(2e-6) == pytest.approx(2e-9)
    assert suggest_diff_step(0.0) == pytest.approx(1e-3)
    assert suggest_diff_step(1e-12, absolute_floor=1e-10) == 1e-10
    with pytest.raises(ValueError, match="relative_step"):
        suggest_diff_step(1.0, relative_step=0.0)
