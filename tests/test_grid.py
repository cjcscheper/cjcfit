"""discrete_grid_fit: the grid point nearest the true minimum, the cost map's layout, and errors."""

import numpy as np
import pytest

from cjcfit import discrete_grid_fit, fit_quality

# x symmetric around 0, so chi^2 = N d_intercept^2 + sum(x^2) d_slope^2 has no cross term:
# the best grid point is then the nearest one in each parameter separately.
X = np.linspace(-5.0, 5.0, 21)


def line(x, slope, intercept):
    """Straight line."""
    return slope * x + intercept


def test_finds_the_grid_point_nearest_the_true_minimum():
    """True (2.03, 0.98) on a 0.1 grid gives (2.0, 1.0), and its chi^2 is the one computed directly."""
    grid = {"slope": np.arange(1.5, 2.55, 0.1), "intercept": np.arange(0.5, 1.55, 0.1)}
    data = line(X, 2.03, 0.98)
    result = discrete_grid_fit(line, X, data, grid, sigma=0.1)
    assert result.best_values == pytest.approx({"slope": 2.0, "intercept": 1.0})
    assert result.chi_squared.shape == (11, 11) and result.parameter_names == ("slope", "intercept")
    direct = np.sum(((data - line(X, **result.best_values)) / 0.1) ** 2)
    assert result.best_chi_squared == pytest.approx(direct)
    # Axis 0 is the slope: chi_squared[i, j] belongs to slope[i], intercept[j].
    assert result.chi_squared[result.best_index] == result.best_chi_squared


def test_complex_data_and_a_model_that_fails_somewhere():
    """Complex data work, and grid points where the model gives NaN get chi^2 = inf."""
    def model(x, scale):
        return np.full(x.shape, np.nan) if scale < 0 else scale * np.exp(1j * x)

    result = discrete_grid_fit(model, X, 0.5 * np.exp(1j * X), {"scale": [-1.0, 0.25, 0.5, 1.0]})
    assert np.isinf(result.chi_squared[0]) and result.best_values == {"scale": 0.5}


@pytest.mark.parametrize(
    ("grid", "message"),
    [({}, "at least one"), ({"slope": []}, "non-empty 1D"), ({"two words": [1.0]}, "identifier")],
)
def test_invalid_grids_raise(grid, message):
    """An empty grid, an empty axis or a name that cannot be a keyword raise a ValueError."""
    with pytest.raises(ValueError, match=message):
        discrete_grid_fit(lambda x, **values: x, X, X, grid)


def test_lowest_chi_squared_is_highest_r_squared_without_sigma():
    """With equal weights R^2 = 1 - chi^2 / sum((data - mean)^2): the best grid point is also the one with the highest R^2."""
    data = line(X, 2.03, 0.98) + 0.3 * np.sin(3 * X)  # a model that cannot fit exactly
    grid = {"slope": np.linspace(1.5, 2.5, 11), "intercept": np.linspace(0.5, 1.5, 11)}
    result = discrete_grid_fit(line, X, data, grid)
    r_squared = np.array([[fit_quality(data, line(X, s, i)).r_squared for i in grid["intercept"]] for s in grid["slope"]])
    assert np.unravel_index(np.argmax(r_squared), r_squared.shape) == result.best_index
    # And R^2 follows chi^2 exactly, since the denominator depends on the data only.
    assert np.allclose(r_squared, 1 - result.chi_squared / np.sum((data - data.mean()) ** 2))
