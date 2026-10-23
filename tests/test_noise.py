"""add_noise and noise_sigma: each form of each term gives the requested spread; errors for impossible settings."""

import numpy as np
import pytest

from cjcfit import add_noise, noise_sigma

N_DRAWS = 40_000
# The standard deviation of a sample standard deviation is sigma / sqrt(2 N): 0.35 % for 40 000 draws.
REL_TOL = 0.02
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def test_additive_number_on_real_data():
    """A number is the standard deviation of every point; the mean stays the signal."""
    noisy = add_noise(np.full(N_DRAWS, 3.0), additive=0.1, seed=1)
    assert np.std(noisy) == pytest.approx(0.1, rel=REL_TOL)
    assert np.mean(noisy) == pytest.approx(3.0, abs=5 * 0.1 / np.sqrt(N_DRAWS))
    assert noisy.dtype == np.float64


def test_additive_array_gives_one_spread_per_point():
    """An array gives each point its own standard deviation (rows are independent draws)."""
    sigma = np.array([0.01, 0.05, 0.1, 0.2, 0.4])
    noisy = add_noise(np.zeros((N_DRAWS, X.size)), additive=sigma, seed=2)
    assert np.allclose(np.std(noisy, axis=0), sigma, rtol=REL_TOL)


def test_additive_function_of_x():
    """A function is evaluated at x and then acts as the array of spreads."""
    noisy = add_noise(np.zeros((N_DRAWS, X.size)), X, additive=lambda x: 0.05 * x, seed=3)
    assert np.allclose(np.std(noisy, axis=0), 0.05 * X, rtol=REL_TOL)


def test_multiplicative_number_on_real_data_scales_with_the_signal():
    """A real multiplicative term is a relative spread: 2 % of 10 is 0.2."""
    noisy = add_noise(np.full(N_DRAWS, 10.0), multiplicative=0.02, seed=4)
    assert np.std(noisy) == pytest.approx(0.2, rel=REL_TOL)


def test_complex_additive_real_sigma_applies_to_both_parts():
    """A real additive sigma on complex data spreads Re and Im by that sigma each; a complex one by its parts."""
    values = np.full(N_DRAWS, 1.0 + 1.0j)
    both = add_noise(values, additive=0.1, seed=5)
    assert np.std(both.real) == pytest.approx(0.1, rel=REL_TOL) and np.std(both.imag) == pytest.approx(0.1, rel=REL_TOL)
    separate = add_noise(values, additive=0.1 + 0.3j, seed=5)
    assert np.std(separate.real) == pytest.approx(0.1, rel=REL_TOL)
    assert np.std(separate.imag) == pytest.approx(0.3, rel=REL_TOL)


def test_complex_multiplicative_gives_amplitude_and_phase_spread():
    """Multiplicative a + ib on complex data: |y| spreads by a relatively, the phase by b radians, exactly."""
    values = np.full(N_DRAWS, 2.0 * np.exp(0.7j))
    noisy = add_noise(values, multiplicative=0.05 + 0.1j, seed=6)
    assert np.std(np.abs(noisy) / 2.0) == pytest.approx(0.05, rel=REL_TOL)
    assert np.std(np.angle(noisy) - 0.7) == pytest.approx(0.1, rel=REL_TOL)
    # A real multiplicative term is amplitude noise only: the phase stays exactly 0.7.
    assert np.allclose(np.angle(add_noise(values, multiplicative=0.05, seed=6)), 0.7)


def test_same_seed_same_noise():
    """The same seed gives identical noise, another seed different noise; a Generator is accepted as seed."""
    values = np.ones(10, dtype=complex)
    first = add_noise(values, additive=0.1, multiplicative=0.01 + 0.02j, seed=42)
    assert np.array_equal(first, add_noise(values, additive=0.1, multiplicative=0.01 + 0.02j, seed=42))
    assert not np.array_equal(first, add_noise(values, additive=0.1, multiplicative=0.01 + 0.02j, seed=43))
    assert np.array_equal(first, add_noise(values, additive=0.1, multiplicative=0.01 + 0.02j, seed=np.random.default_rng(42)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"additive": 0.1j}, r"additive noise has an imaginary part \(noise on Im\(y\)\)"),
        ({"multiplicative": 0.01 + 0.01j}, r"imaginary part \(phase noise\)"),
        ({"additive": -0.1}, "non-negative"),
        ({"additive": np.ones(3)}, "does not match"),
        ({"additive": lambda x: x}, "x must be given"),
    ],
)
def test_impossible_settings_raise_clear_errors(kwargs, message):
    """Imaginary parts on real data, negative spreads, wrong shapes and functions without x raise a ValueError."""
    with pytest.raises(ValueError, match=message):
        add_noise(np.ones(5), **kwargs)


def test_noise_sigma_matches_the_drawn_spread():
    """noise_sigma gives the measured spread of Re and Im (complex) and of the values (real)."""
    values = np.full(N_DRAWS, 2.0 * np.exp(0.7j))
    settings = {"additive": 0.01 + 0.02j, "multiplicative": 0.03 + 0.05j}
    noisy = add_noise(values, **settings, seed=7)
    sigma = noise_sigma(values, **settings)
    # First order in a and b: good to about a percent for these small spreads.
    assert np.std(noisy.real) == pytest.approx(sigma[0].real, rel=0.03)
    assert np.std(noisy.imag) == pytest.approx(sigma[0].imag, rel=0.03)
    # Real data: sqrt((a y)^2 + s^2) = sqrt(0.3^2 + 0.4^2) = 0.5 for y = 10, a = 0.03, s = 0.4.
    assert np.allclose(noise_sigma(np.full(4, 10.0), additive=0.4, multiplicative=0.03), 0.5)
