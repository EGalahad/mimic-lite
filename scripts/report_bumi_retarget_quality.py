#!/usr/bin/env python3
# ruff: noqa: I001
"""Compatibility CLI for GMR's MimicLite retarget quality report."""

from mimic_lite_conversion._gmr_compat import ensure_gmr_on_path

ensure_gmr_on_path()

from general_motion_retargeting.integrations.mimic_lite.quality import *


if __name__ == "__main__":
    main()
