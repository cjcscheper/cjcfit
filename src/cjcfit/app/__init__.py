"""Everything in cjcfit that needs a screen: the interactive start-point picker.

``pick_initial_values()`` shows the chi-squared map of
:func:`cjcfit.discrete_grid_fit` over two parameters, lets you click a start
point while the model curve at that point is drawn over the data, and returns
the values chosen when the window is closed.

Needs the ``app`` extra (matplotlib and cjcplot): ``pip install "cjcfit[app]"``.
The core package (``import cjcfit``) works without it.
"""

# Check the optional dependencies first, so a missing one gives an install hint instead of a bare ImportError.
try:
    import cjcplot  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError as error:
    raise ImportError(
        f"cjcfit.app needs matplotlib and cjcplot ({error.name} is missing). "
        'Install them with: pip install "cjcfit[app] @ git+https://github.com/cjcscheper/cjcfit.git" '
        '(or pip install -e ".[app]" in a clone).'
    ) from error

from cjcfit.app.grid_window import pick_initial_values

__all__ = ["pick_initial_values"]
