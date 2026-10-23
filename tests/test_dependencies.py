"""The core needs only numpy and scipy; the app says how to install the rest. Runs with or without the app extra."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def test_core_does_not_import_matplotlib_or_cjcplot():
    """`import cjcfit` loads neither matplotlib nor cjcplot (checked in a fresh interpreter)."""
    code = "import sys, cjcfit; print(sorted(m for m in ('matplotlib', 'cjcplot') if m in sys.modules))"
    # The fresh interpreter finds the package through PYTHONPATH, as pytest's own pythonpath setting does not reach it.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == "[]"


def test_app_without_the_extra_explains_how_to_install_it(monkeypatch):
    """Importing cjcfit.app without cjcplot raises an ImportError that names the app extra."""
    # Pretend cjcplot is not installed (None in sys.modules makes `import cjcplot` fail) and forget any
    # cjcfit.app modules imported by earlier tests, so the import really runs again; monkeypatch restores both.
    monkeypatch.setitem(sys.modules, "cjcplot", None)
    for name in [name for name in sys.modules if name.startswith("cjcfit.app")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r"cjcfit\[app\]"):
        importlib.import_module("cjcfit.app")
