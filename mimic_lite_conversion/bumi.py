"""Compatibility imports for the GMR-owned MimicLite Bumi exporter."""

from ._gmr_compat import ensure_gmr_on_path

ensure_gmr_on_path()

from general_motion_retargeting.integrations.mimic_lite.bumi import *
