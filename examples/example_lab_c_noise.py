"""Reference: the measured noise of the Lab C THz-TDS setup (calibration of 2026-06-15) as cjcfit noise terms.

The calibration describes the noise of a complex field ratio t(f) = E_sample / E_reference with

- an additive term B(f), the "bathtub": large below ~0.3 THz (1/f excess of the low-frequency end) and above
  ~4 THz (where the reference spectrum sinks into the noise floor), small in between;
- a multiplicative term with a flat relative amplitude noise (laser intensity noise, 1.8e-4) and a phase noise
  that grows linearly with frequency, 2 pi f sigma_tau, from a timing jitter sigma_tau = 4.80 fs.

The numbers below are from that calibration (dark-current and air-reference traces) and belong to that setup
and date only: copy this block into a script that needs them, and replace it after a new calibration.
The additive term acts on Re and Im of t, the multiplicative one on its amplitude and phase, as in every
cjcfit noise model.
"""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors

from cjcfit import add_noise, noise_sigma

output_stem = Path(__file__).with_suffix("")
BLUE, YELLOW, RED = style_colors()[:3]

# --- Lab C calibration, 2026-06-15 ---------------------------------------------------------------------------
# Additive bathtub B(f) = A_exp exp(-f / f_exp) + A_tanh (1 + tanh((f - f0) / w)) / 2 + C, f in THz.
BATHTUB_A_EXP = 4.046337e-3
BATHTUB_F_EXP_THZ = 1.037124e-1
BATHTUB_A_TANH = 2.826672e-1
BATHTUB_F0_THZ = 5.141837e0
BATHTUB_W_THZ = 1.254768e0
BATHTUB_C = 4.819142e-20
# Multiplicative: relative amplitude noise, and the phase-noise slope 2 pi 1e12 sigma_tau (sigma_tau = 4.80 fs).
AMPLITUDE_NOISE = 1.823889e-4
PHASE_NOISE_SLOPE_RAD_PER_THZ = 3.018127e-2


def lab_c_additive(freq_hz):
    """Standard deviation of Re(t) and of Im(t) (a real number: the same for both), from the bathtub B(f)."""
    freq_thz = freq_hz * 1e-12
    return (
        BATHTUB_A_EXP * np.exp(-freq_thz / BATHTUB_F_EXP_THZ)
        + BATHTUB_A_TANH * 0.5 * (1.0 + np.tanh((freq_thz - BATHTUB_F0_THZ) / BATHTUB_W_THZ))
        + BATHTUB_C
    )


def lab_c_multiplicative(freq_hz):
    """Relative amplitude noise (real part) and phase noise in rad (imaginary part) of t(f)."""
    return AMPLITUDE_NOISE + 1j * PHASE_NOISE_SLOPE_RAD_PER_THZ * freq_hz * 1e-12
# --------------------------------------------------------------------------------------------------------------


# A synthetic field ratio to put the noise on: a sample that attenuates more at high frequency and delays by 2.4 ps.
freq_hz = np.linspace(0.1e12, 4.5e12, 89)
freq_thz = freq_hz * 1e-12
field_ratio = 0.7 * np.exp(-freq_thz / 3.0) * np.exp(-2j * np.pi * freq_hz * 2.4e-12)

# The noise terms as functions of frequency: add_noise evaluates them at the frequencies it is given.
noisy = add_noise(field_ratio, freq_hz, additive=lab_c_additive, multiplicative=lab_c_multiplicative, seed=6)
sigma = noise_sigma(field_ratio, freq_hz, additive=lab_c_additive, multiplicative=lab_c_multiplicative)

# The additive bathtub on a log scale.
freq_fine_hz = np.linspace(0.05e12, 6.0e12, 600)
plot_1d(
    freq_fine_hz * 1e-12, lab_c_additive(freq_fine_hz),
    colors=[BLUE],
    legend_labels=[r"additive $B(f)$"],
    xlabel="Frequency [THz]", ylabel=r"$\sigma$",
    xlim=(0.0, 6.0), ylim=(1e-4, 1.0),
    logy=True,
    legend_loc="lower right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_additive.png",
    fig_show=False,
)

# The multiplicative terms: flat amplitude noise and the phase noise of the timing jitter.
multiplicative = lab_c_multiplicative(freq_fine_hz)
plot_1d(
    [freq_fine_hz * 1e-12] * 2, [multiplicative.real, multiplicative.imag],
    colors=[BLUE, RED],
    legend_labels=["amplitude", "phase"],
    xlabel="Frequency [THz]", ylabel=r"$\sigma$ [relative, rad]",
    xlim=(0.0, 6.0), ylim=(1e-5, 1.0),
    logy=True,
    legend_loc="upper left",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_multiplicative.png",
    fig_show=False,
)

# |t| of one noisy realisation; its sigma is about the larger of the Re and Im sigma.
fine_ratio = 0.7 * np.exp(-freq_fine_hz * 1e-12 / 3.0)
plot_1d(
    [freq_thz, freq_fine_hz * 1e-12], [np.abs(noisy), fine_ratio],
    yerr_data=[np.maximum(sigma.real, sigma.imag), None],
    colors=[BLUE, RED], err_colors=[BLUE],
    line_show=[False, True], scatter_show=[True, False], line_styles=["-", "-"],
    legend_labels=["noisy", "noise-free"],
    xlabel="Frequency [THz]", ylabel=r"$|t(f)|$",
    xlim=(0.0, 4.6), ylim=(0.0, 0.9),
    legend_loc="upper right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_amplitude.png",
    fig_show=False,
)

# Phase of the same realisation, minus the delay's ramp so the noise is visible: about sqrt(b^2 + (B / |t|)^2).
ramp = 2 * np.pi * freq_hz * 2.4e-12
phase_sigma = np.sqrt(lab_c_multiplicative(freq_hz).imag ** 2 + (lab_c_additive(freq_hz) / np.abs(field_ratio)) ** 2)
plot_1d(
    [freq_thz, freq_thz], [np.unwrap(np.angle(noisy)) + ramp, np.zeros(freq_hz.size)],
    yerr_data=[phase_sigma, None],
    colors=[BLUE, RED], err_colors=[BLUE],
    line_show=[False, True], scatter_show=[True, False], line_styles=["-", "-"],
    legend_labels=["noisy", "noise-free"],
    xlabel="Frequency [THz]", ylabel=r"Phase of $t$ + $2\pi f \tau$ [rad]",
    xlim=(0.0, 4.6), ylim=(-1.0, 1.0),
    legend_loc="upper left",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_phase.png",
    fig_show=False,
)
