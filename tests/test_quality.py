"""fit_quality with residuals that can be worked out by hand."""

import numpy as np
import pytest

from cjcfit import fit_quality

MODEL = np.sin(np.linspace(0.0, 6.0, 50))


def test_identical_curves_are_a_perfect_fit():
    """Model equal to the data: chi^2 0, R^2 1, overlap 1, scale factor 1."""
    quality = fit_quality(MODEL, MODEL, sigma=0.1, n_parameters=2)
    assert quality.chi_squared == 0.0 and quality.r_squared == pytest.approx(1.0)
    assert np.allclose(quality.overlap, 1.0) and quality.overall_overlap == pytest.approx(1.0)
    assert quality.scale_factor == pytest.approx(1.0) and quality.degrees_of_freedom == 48


def test_constant_offset():
    """An offset of 0.1 with sigma 0.05: every normalized residual is 2, so chi^2 = 4 N."""
    quality = fit_quality(MODEL + 0.1, MODEL, sigma=0.05, n_parameters=1)
    assert np.allclose(quality.normalized_residuals, 2.0) and np.allclose(quality.residuals, 0.1)
    assert quality.chi_squared == pytest.approx(4 * 50)
    assert quality.reduced_chi_squared == pytest.approx(4 * 50 / 49)
    assert quality.mean_normalized_residual == pytest.approx(2.0) and quality.std_normalized_residual == pytest.approx(0.0, abs=1e-12)
    assert quality.rms_residual == pytest.approx(0.1) and quality.max_abs_normalized_residual == pytest.approx(2.0)


def test_complex_data_count_real_and_imaginary_parts():
    """Complex data: Re and Im residuals each count in chi^2, and a factor 2j shows as the scale factor."""
    model = np.exp(1j * np.linspace(0.0, 6.0, 50))
    quality = fit_quality(model + (0.1 + 0.2j), model, sigma=0.1 + 0.1j)
    # Normalized residual 1 + 2i per point: 1^2 + 2^2 = 5 per point.
    assert quality.chi_squared == pytest.approx(5 * 50) and quality.degrees_of_freedom == 100
    assert fit_quality(2j * model, model).scale_factor == pytest.approx(2j)


def test_r_squared_of_the_mean_is_zero():
    """A model equal to the mean of the data explains nothing: R^2 = 0."""
    assert fit_quality(MODEL, np.full_like(MODEL, MODEL.mean())).r_squared == pytest.approx(0.0, abs=1e-9)


def test_mismatched_shapes_raise():
    """A model curve of another shape than the data raises a ValueError."""
    with pytest.raises(ValueError, match="shape"):
        fit_quality(MODEL, MODEL[:10])
