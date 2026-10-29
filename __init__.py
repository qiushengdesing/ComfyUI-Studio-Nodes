"""
ComfyUI-Studio-Nodes: High Performance Image Enhancement & Utility Nodes for ComfyUI.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 1. Load ERP & Seam Tools
try:
    from .erp_tools import NODE_CLASS_MAPPINGS as ERP_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as ERP_NAMES
    NODE_CLASS_MAPPINGS.update(ERP_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(ERP_NAMES)
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] erp_tools notice: {e}")

# 2. Load DiT360 Projection & Viewer
try:
    from .dit360 import NODE_CLASS_MAPPINGS as DIT_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as DIT_NAMES
    NODE_CLASS_MAPPINGS.update(DIT_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(DIT_NAMES)
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] dit360 notice: {e}")

# 3. Load Klein Enhancer & Multi-reference Latents
try:
    from .klein_enhancer import NODE_CLASS_MAPPINGS as KLEIN_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as KLEIN_NAMES
    NODE_CLASS_MAPPINGS.update(KLEIN_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(KLEIN_NAMES)
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] klein_enhancer notice: {e}")

# 4. Load Flux2 Primitives Compatibility (includes ReferenceLatent & ColorTransfer fallbacks)
try:
    from .primitives import NODE_CLASS_MAPPINGS as PRIM_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as PRIM_NAMES
    # Only add if not already registered or override cleanly
    for k, v in PRIM_CLASSES.items():
        if k not in NODE_CLASS_MAPPINGS:
            NODE_CLASS_MAPPINGS[k] = v
            NODE_DISPLAY_NAME_MAPPINGS[k] = PRIM_NAMES.get(k, k)
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] primitives notice: {e}")

# 5. Load SeedVR2 Upscaler nodes for classic ComfyUI mappings
try:
    from .seedvr2.src.interfaces.video_upscaler import SeedVR2VideoUpscaler
    from .seedvr2.src.interfaces.dit_model_loader import SeedVR2LoadDiTModel
    from .seedvr2.src.interfaces.vae_model_loader import SeedVR2LoadVAEModel
    
    NODE_CLASS_MAPPINGS["SeedVR2VideoUpscaler"] = SeedVR2VideoUpscaler
    NODE_CLASS_MAPPINGS["SeedVR2LoadDiTModel"] = SeedVR2LoadDiTModel
    NODE_CLASS_MAPPINGS["SeedVR2LoadVAEModel"] = SeedVR2LoadVAEModel
    
    NODE_DISPLAY_NAME_MAPPINGS["SeedVR2VideoUpscaler"] = "SeedVR2 Video Upscaler"
    NODE_DISPLAY_NAME_MAPPINGS["SeedVR2LoadDiTModel"] = "SeedVR2 Load DiT Model"
    NODE_DISPLAY_NAME_MAPPINGS["SeedVR2LoadVAEModel"] = "SeedVR2 Load VAE Model"
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] seedvr2 notice: {e}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
