"""Locate the local GMR checkout for deprecated MimicLite import paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_gmr_on_path() -> Path:
    """Add GMR to ``sys.path`` and return its repository root.

    ``GMR_ROOT`` is the portable override.  The sibling-workspace search keeps
    the existing EGalahad/UniLab development layout zero-configuration.
    """

    candidates: list[Path] = []
    configured = os.environ.get("GMR_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())

    source = Path(__file__).resolve()
    for parent in source.parents:
        candidates.append(parent / "UniLab" / "thirdparty" / "GMR")

    for candidate in candidates:
        root = candidate.resolve()
        package = root / "general_motion_retargeting"
        if (package / "__init__.py").is_file():
            root_string = str(root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            return root

    raise ModuleNotFoundError(
        "Cannot locate the GMR checkout. Set GMR_ROOT to the GMR repository "
        "before importing mimic_lite_conversion."
    )
