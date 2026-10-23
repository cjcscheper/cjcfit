"""One general noise model with two terms, for pseudo-measurements and Monte Carlo studies.

:func:`add_noise` draws a noisy copy of a curve; :func:`noise_sigma` gives the
standard deviation per point that the same settings produce, in the form
:func:`cjcfit.fit_least_squares` takes as ``sigma``.

The model (THEORY.md, section 8)::

    y_noisy = y * (1 + a * n1) * exp(i * b * n2)  +  (s_re * n3 + i * s_im * n4)

with independent standard normal numbers n1..n4 per point, and

- ``multiplicative = a + i b``: ``a`` the relative amplitude noise, ``b`` the
  phase noise in radians, applied exactly (a rotation, not linearised);
- ``additive = s_re + i s_im``: noise independent of the signal, ``s_re`` on
  Re(y) and ``s_im`` on Im(y). A *real* additive term on complex data applies
  to both parts (``s_re = s_im``).

Each term may be a number, an array with one value per point (noise that
depends on x, e.g. on frequency), or a function of x returning such an array.
For real data only the real parts exist: an imaginary part there is an error.
"""

from __future__ import annotations

import numpy as np


def add_noise(
    values, x=None, *, additive=0.0, multiplicative=0.0, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """A noisy copy of ``values`` with the two-term noise model of this module.

    Parameters
    ----------
    values : array-like
        The noise-free curve, real or complex, any shape.
    x : anything, optional
        Passed to ``additive`` and ``multiplicative`` when they are functions;
        needed only then.
    additive : number, array-like or callable, default 0
        Signal-independent noise: real part = standard deviation on Re(y),
        imaginary part = on Im(y). A real value on complex data applies to
        both parts. A callable is called as ``additive(x)``.
    multiplicative : number, array-like or callable, default 0
        Signal-proportional noise: real part = relative amplitude standard
        deviation (0.01 = 1 %), imaginary part = phase standard deviation in
        radians. A real value is amplitude noise only. Applied as
        ``y (1 + a n1) exp(i b n2)``; keep ``a`` well below 1, or
        ``1 + a n1`` can become negative.
    seed : int, numpy.random.Generator or None
        The same seed gives the same noise. None draws fresh noise each call.

    Returns
    -------
    ndarray
        Same shape as ``values``; real for real values, complex for complex.

    Raises
    ------
    ValueError
        For a negative standard deviation, a term that does not match the shape
        of ``values``, an imaginary part on real data, or a function without x.
    """
    # The two terms as real and imaginary standard deviations per point, checked against the data.
    values = np.asarray(values)
    is_complex = np.iscomplexobj(values)
    amplitude_std, phase_std = _term_parts(multiplicative, x, values.shape, "multiplicative", is_complex, False)
    additive_re_std, additive_im_std = _term_parts(additive, x, values.shape, "additive", is_complex, True)

    # Always four normal arrays in this order, so the noise of one term does not depend on the other being zero.
    generator = np.random.default_rng(seed)
    n1, n2, n3, n4 = generator.standard_normal((4, *values.shape))

    # Real data: amplitude factor and additive noise only.
    if not is_complex:
        return values.astype(np.float64) * (1.0 + amplitude_std * n1) + additive_re_std * n3

    # Complex data: exact amplitude factor and phase rotation, then the additive noise on each part.
    noisy = values.astype(np.complex128) * (1.0 + amplitude_std * n1) * np.exp(1j * phase_std * n2)
    return noisy + additive_re_std * n3 + 1j * additive_im_std * n4


def noise_sigma(values, x=None, *, additive=0.0, multiplicative=0.0) -> np.ndarray:
    """Standard deviation per point that :func:`add_noise` gives with the same settings, as a fit's ``sigma``.

    For real data: ``sqrt((a y)² + s_re²)``. For complex data the standard
    deviations of Re and Im, returned as ``sigma_re + 1j sigma_im``, to first
    order in ``a`` and ``b`` (``d y = y (a n1 + i b n2)``)::

        sigma_re² = (a Re y)² + (b Im y)² + s_re²
        sigma_im² = (a Im y)² + (b Re y)² + s_im²

    Where ``a != b`` the Re and Im noise of one point are correlated, which a
    per-point sigma cannot express; the fit still weights each point sensibly.

    Parameters and errors as in :func:`add_noise`.
    """
    # The same term parsing as add_noise, so both always agree.
    values = np.asarray(values)
    is_complex = np.iscomplexobj(values)
    amplitude_std, phase_std = _term_parts(multiplicative, x, values.shape, "multiplicative", is_complex, False)
    additive_re_std, additive_im_std = _term_parts(additive, x, values.shape, "additive", is_complex, True)

    # Real data: amplitude and additive noise add in quadrature.
    if not is_complex:
        return np.sqrt((amplitude_std * values) ** 2 + additive_re_std**2) * np.ones(values.shape)

    # Complex data: the linearised spread of Re and Im (see the docstring), plus the additive parts.
    real, imag = values.real, values.imag
    sigma_re = np.sqrt((amplitude_std * real) ** 2 + (phase_std * imag) ** 2 + additive_re_std**2)
    sigma_im = np.sqrt((amplitude_std * imag) ** 2 + (phase_std * real) ** 2 + additive_im_std**2)
    return sigma_re + 1j * sigma_im


def _term_parts(term, x, shape, name: str, data_is_complex: bool, real_means_both: bool):
    """One noise term as (real-part std, imaginary-part std) arrays of the data's shape.

    ``real_means_both``: a real additive term means the same std on Re and Im;
    a real multiplicative term means amplitude noise only (no phase noise).
    """
    # A function is evaluated at x first; it then counts as an array.
    if callable(term):
        if x is None:
            raise ValueError(f"The {name} noise is a function of x, so x must be given.")
        term = term(x)
    term = np.asarray(term)
    try:
        term = np.broadcast_to(term, shape)
    except ValueError:
        raise ValueError(f"The {name} noise of shape {term.shape} does not match the values of shape {shape}.") from None

    # Split into the two parts; a real term is "both" (additive) or "amplitude only" (multiplicative).
    if np.iscomplexobj(term):
        real_part, imag_part = term.real.astype(np.float64), term.imag.astype(np.float64)
    else:
        real_part = term.astype(np.float64)
        imag_part = real_part if real_means_both else np.zeros(shape)

    # Real data have no imaginary part (and no phase) to put noise on.
    if not data_is_complex and np.iscomplexobj(term) and np.any(term.imag != 0):
        meaning = "noise on Im(y)" if real_means_both else "phase noise"
        raise ValueError(f"The data are real, but the {name} noise has an imaginary part ({meaning}).")

    # Standard deviations cannot be negative.
    if np.any(real_part < 0) or np.any(imag_part < 0) or not (np.all(np.isfinite(real_part)) and np.all(np.isfinite(imag_part))):
        raise ValueError(f"The {name} noise must be finite and non-negative in both parts.")
    return real_part, imag_part
