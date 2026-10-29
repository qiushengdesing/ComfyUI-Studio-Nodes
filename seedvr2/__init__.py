"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

# Compatibility shim for Transformers 5.5+.
# Some releases query the flash-attn distribution mapping before registering it,
# which raises KeyError on systems where flash-attn is intentionally not installed.
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    if "flash_attn" not in PACKAGE_DISTRIBUTION_MAPPING:
        PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] = ["flash_attn", "flash-attn"]
except Exception:
    pass

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import comfy_entrypoint, SeedVR2Extension

__all__ = ["comfy_entrypoint", "SeedVR2Extension"]
