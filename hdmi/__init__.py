"""Backward-compatible alias for the renamed :mod:`mimic_lite` package."""

from mimic_lite import *  # noqa: F401,F403

import sys as _sys

import mimic_lite as _mimic_lite

__path__ = _mimic_lite.__path__

for _name, _module in list(_sys.modules.items()):
    if _name == "mimic_lite" or _name.startswith("mimic_lite."):
        _sys.modules.setdefault(_name.replace("mimic_lite", "hdmi", 1), _module)
