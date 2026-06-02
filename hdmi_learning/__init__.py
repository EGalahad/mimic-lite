"""Backward-compatible alias for the renamed :mod:`mimic_lite_learning` package."""

from mimic_lite_learning import *  # noqa: F401,F403

import sys as _sys

import mimic_lite_learning as _mimic_lite_learning

__path__ = _mimic_lite_learning.__path__

for _name, _module in list(_sys.modules.items()):
    if _name == "mimic_lite_learning" or _name.startswith("mimic_lite_learning."):
        _sys.modules.setdefault(
            _name.replace("mimic_lite_learning", "hdmi_learning", 1),
            _module,
        )
