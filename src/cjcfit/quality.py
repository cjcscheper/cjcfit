"""How well a model curve describes data: chi-squared, residual statistics, R², overlap.

:func:`fit_quality` compares any model curve with data, from a fit or not
(e.g. a model with literature values against a measurement).
:func:`cjcfit.fit_least_squares` and :func:`cjcfit.discrete_grid_fit` use the
same residual definition (the private helpers below), so chi-squared means
the same thing everywhere in the package.

Residuals are ``data - model``. With ``sigma`` they are divided by it
("normalized residuals"); complex data are compared part by part,
``Re(r)/sigma_re + i Im(r)/sigma_im`` (THEORY.md, section 7). ``weights``
multiply the residuals the fit minimises; a weight 0 leaves a point out
(THEORY.md, section 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitQuality:
    """Agreement between data and a model curve, from :func:`fit_quality`.

    Attributes
    ----------
    residuals : ndarray
        ``data - model_curve``, shape and type (real or complex) of the data.
    normalized_residuals : ndarray
        Residuals divided by sigma, part by part for complex data
        (``Re/sigma_re + i Im/sigma_im``). Without sigma but with weights,
        residuals times weights (weights taken as relative ``1 / sigma``);
        with neither equal to ``residuals``.
    chi_squared : float
        Sum of the squared normalized residuals (Re and Im both count) over
        the points with a non-zero weight.
    degrees_of_freedom : int
        Number of real data values with a non-zero weight (2 per complex
        point) minus ``n_parameters``.
    reduced_chi_squared : float
        ``chi_squared / degrees_of_freedom`` (NaN for no degrees of freedom).
        About 1 when the model describes the data and sigma is right; well
        above 1 means a wrong model or a too small sigma, well below 1 a too
        large sigma. Only meaningful with sigma.
    r_squared : float
        ``1 - sum|r|² / sum|data - mean(data)|²``: the fraction of the data's
        variation the model explains (1 = perfect, 0 = no better than the mean).
        Without sigma or weights inside, so it can be compared between fits
        with different sigma; points with weight 0 are left out (Re and Im
        each around their own mean).
    rms_residual : float
        ``sqrt(mean |r|²)``, in the units of the data, over all points.
    mean_normalized_residual, std_normalized_residual : float
        Mean and standard deviation of the normalized residuals (Re and Im
        values pooled, points with weight 0 left out). About 0 and 1 for a
        good fit with the right sigma; a mean far from 0 shows a systematic offset.
    max_abs_normalized_residual : float
        Largest ``|normalized residual|`` of a single real value: outliers.
    overlap : ndarray
        Per point ``1 - |r| / (|data| + |model| + eps)``: 1 where model and
        data agree, 0 where they are opposite or one of them is zero.
    overall_overlap : float
        ``1 / (1 + ||r|| / (||data|| + ||model|| + eps))`` with L2 norms: one
        number between 0.5 (completely different) and 1 (identical).
    scale_factor : complex
        Best complex factor ``g`` in ``data - mean ≈ g (model - mean)``: about 1
        when the model has the right size; a different value shows a missing
        calibration factor (its phase, a constant phase offset).
    """

    residuals: np.ndarray
    normalized_residuals: np.ndarray
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    r_squared: float
    rms_residual: float
    mean_normalized_residual: float
    std_normalized_residual: float
    max_abs_normalized_residual: float
    overlap: np.ndarray
    overall_overlap: float
    scale_factor: complex


def fit_quality(
    data, model_curve, *, sigma=None, weights=None, n_parameters: int = 0, eps: float = 1e-12
) -> FitQuality:
    """Compare ``model_curve`` with ``data`` and return a :class:`FitQuality`.

    Parameters
    ----------
    data, model_curve : array-like
        Same shape; real or complex. Interpolate the model onto the data's
        x-values and cut both to the range of interest before calling.
    sigma : float or array-like, optional
        Standard deviation per point, as in :func:`cjcfit.fit_least_squares`:
        real sigma applies to both Re and Im of complex data, complex sigma
        gives ``sigma_re + i sigma_im``. None: residuals are not normalized.
    weights : float or array-like, optional
        As in :func:`cjcfit.fit_least_squares`: points with weight 0 are left
        out of chi-squared and the residual statistics; without sigma the
        weights act as relative ``1 / sigma``.
    n_parameters : int, default 0
        Number of fitted parameters, subtracted for the degrees of freedom.
    eps : float, default 1e-12
        Small number in the denominators of R², overlap and scale factor, so
        an all-zero curve gives a number instead of a division by zero.
    """
    # Same shapes; a model curve for real data must itself be real.
    data, model_curve = _matching_data_and_model(data, model_curve)
    if not eps > 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    # The residuals, raw and normalized, and chi-squared over the points that count.
    residuals = data - model_curve
    scales = _residual_scales(data, sigma, weights)
    stacked = _stack_real_imag(residuals) * scales.chi_factor
    normalized_residuals = _unstack(stacked, data)
    counted = stacked[scales.included]
    chi_squared = float(np.sum(counted**2))
    degrees_of_freedom = counted.size - int(n_parameters)
    reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

    # R² against the data's own mean, without sigma or weights, over the points that count.
    r_squared = _r_squared(data, model_curve, scales.included, eps)

    # Overlap per point and as one number from the L2 norms.
    overlap = 1.0 - np.abs(residuals) / (np.abs(data) + np.abs(model_curve) + eps)
    norm_residuals, norm_data, norm_model = (float(np.linalg.norm(a)) for a in (residuals, data, model_curve))
    overall_overlap = 1.0 / (1.0 + norm_residuals / (norm_data + norm_model + eps))

    # Least-squares factor between the mean-free curves (np.vdot conjugates the model).
    model_centred = model_curve - np.mean(model_curve)
    data_centred = data - np.mean(data)
    scale_factor = complex(np.vdot(model_centred, data_centred) / (np.vdot(model_centred, model_centred) + eps))

    return FitQuality(
        residuals=residuals,
        normalized_residuals=normalized_residuals,
        chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=float(reduced_chi_squared),
        r_squared=r_squared,
        rms_residual=float(np.sqrt(np.mean(np.abs(residuals) ** 2))),
        mean_normalized_residual=float(np.mean(counted)),
        std_normalized_residual=float(np.std(counted, ddof=1)) if counted.size > 1 else np.nan,
        max_abs_normalized_residual=float(np.max(np.abs(counted))),
        overlap=overlap,
        overall_overlap=overall_overlap,
        scale_factor=scale_factor,
    )


@dataclass(frozen=True)
class _ResidualScales:
    """Factors per stacked real residual value (Re..., Im...): one place for sigma and weights.

    fit_factor : what the optimiser minimises the squares of is ``fit_factor * (data - model)``,
        i.e. ``weight / sigma`` (sigma = 1 and weight = 1 when not given).
    chi_factor : the normalized residual is ``chi_factor * (data - model)``: ``1 / sigma`` with sigma,
        else ``weight`` (weights as relative 1 / sigma), else 1.
    included : points with a non-zero weight, the ones chi-squared and the statistics count.
    weights : the stacked weights (ones without weights), for the covariance with sigma and weights.
    has_sigma : whether sigma was given (absolute uncertainties) or not (scaled by the scatter).
    """

    fit_factor: np.ndarray
    chi_factor: np.ndarray
    included: np.ndarray
    weights: np.ndarray
    has_sigma: bool


def _matching_data_and_model(data, model_curve) -> tuple[np.ndarray, np.ndarray]:
    """Data and model curve as arrays of one shape; a complex model for real data must have zero imaginary part."""
    # Shapes must agree exactly: broadcasting would hide a model that returns the wrong length.
    data = np.asarray(data)
    model_curve = np.asarray(model_curve)
    if model_curve.shape != data.shape:
        raise ValueError(f"The model returned shape {model_curve.shape}, but the data have shape {data.shape}.")

    # Real data: a complex model with a non-zero imaginary part cannot describe them.
    if not np.iscomplexobj(data):
        if np.iscomplexobj(model_curve):
            if np.any(model_curve.imag != 0):
                raise ValueError("The data are real, but the model returned complex values with a non-zero imaginary part.")
            model_curve = model_curve.real
        return data.astype(np.float64), model_curve.astype(np.float64)
    return data.astype(np.complex128), model_curve.astype(np.complex128)


def _residual_scales(data: np.ndarray, sigma, weights) -> _ResidualScales:
    """The factors that turn stacked raw residuals into fit and normalized residuals (see :class:`_ResidualScales`)."""
    # Sigma must be positive (a division), weights may be 0 (a point left out) but not all of them.
    sigma_stacked = _stacked_parts(sigma, data, "sigma", zero_allowed=False)
    weights_stacked = _stacked_parts(weights, data, "weights", zero_allowed=True)
    if weights_stacked is not None and not np.any(weights_stacked > 0):
        raise ValueError("At least one weight must be larger than 0.")

    # Missing ones count as 1; the normalized residual follows sigma if given, else the weights.
    size = 2 * data.size if np.iscomplexobj(data) else data.size
    ones = np.ones(size)
    sigma_values = ones if sigma_stacked is None else sigma_stacked
    weight_values = ones if weights_stacked is None else weights_stacked
    chi_factor = 1.0 / sigma_values if sigma_stacked is not None else weight_values
    return _ResidualScales(
        fit_factor=weight_values / sigma_values,
        chi_factor=chi_factor,
        included=weight_values > 0,
        weights=weight_values,
        has_sigma=sigma_stacked is not None,
    )


def _r_squared(data: np.ndarray, model_curve: np.ndarray, included: np.ndarray, eps: float = 1e-12) -> float:
    """R² over the stacked real values that count: ``1 - sum(residual²) / sum((data - mean)²)``.

    For complex data Re and Im are each measured from their own mean, so the
    denominator is ``sum|data - mean(data)|²`` (the same as for one complex
    number per point) when no point is left out.
    """
    # Stacked data and residuals; the mean of each part is taken over the values that count.
    data_stacked = _stack_real_imag(data)
    residuals = data_stacked - _stack_real_imag(model_curve)
    centred = data_stacked.copy()
    half = data.size
    for part in ((slice(0, half), slice(half, 2 * half)) if np.iscomplexobj(data) else (slice(None),)):
        if np.any(included[part]):
            centred[part] -= np.mean(data_stacked[part][included[part]])

    # Fraction of the variation that is left in the residuals.
    sum_squared_residuals = float(np.sum(residuals[included] ** 2))
    sum_squared_variation = float(np.sum(centred[included] ** 2))
    return 1.0 - sum_squared_residuals / (sum_squared_variation + eps)


def _stacked_parts(values, data: np.ndarray, name: str, zero_allowed: bool) -> np.ndarray | None:
    """``sigma`` or ``weights`` as one stacked real vector matching :func:`_stack_real_imag` of the data.

    A real value applies to both Re and Im of complex data; a complex value
    gives the Re part and the Im part separately. Real data refuse a non-zero
    imaginary part. None stays None.
    """
    # Nothing given: nothing to stack.
    if values is None:
        return None
    values = np.asarray(values)
    try:
        values = np.broadcast_to(values, data.shape)
    except ValueError:
        raise ValueError(f"{name} of shape {values.shape} does not match the data of shape {data.shape}.") from None

    # Real data have no imaginary part to put a sigma or weight on.
    if not np.iscomplexobj(data):
        if np.iscomplexobj(values) and np.any(values.imag != 0):
            raise ValueError(f"The data are real, so {name} must be real (it has a non-zero imaginary part).")
        stacked = np.ravel(values.real).astype(np.float64)
    elif np.iscomplexobj(values):
        stacked = np.concatenate([np.ravel(values.real), np.ravel(values.imag)]).astype(np.float64)
    else:
        stacked = np.tile(np.ravel(values).astype(np.float64), 2)

    # Sigma divides (positive); a weight multiplies (zero or positive).
    valid = (stacked >= 0) if zero_allowed else (stacked > 0)
    if not np.all(np.isfinite(stacked) & valid):
        need = "zero or positive" if zero_allowed else "positive"
        raise ValueError(f"{name} must be {need} and finite at every point (for complex {name}: both parts).")
    return stacked


def _stack_real_imag(values: np.ndarray) -> np.ndarray:
    """Flatten to one real vector; complex values become [Re..., Im...] (THEORY.md, section 7)."""
    # Least squares needs real numbers; a complex point is two real measurements.
    values = np.ravel(values)
    if np.iscomplexobj(values):
        return np.concatenate([values.real, values.imag])
    return values.astype(np.float64)


def _unstack(stacked: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_stack_real_imag`: back to the shape and type (real or complex) of ``like``."""
    if np.iscomplexobj(like):
        half = stacked.size // 2
        return (stacked[:half] + 1j * stacked[half:]).reshape(like.shape)
    return stacked.reshape(like.shape)
