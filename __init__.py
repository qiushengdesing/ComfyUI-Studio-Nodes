"""
ComfyUI-Studio-Nodes: 360 ERP Panorama Seam Healing, Shaniu 6.0 Algorithm,
Polar Structure Guard, and Panorama Workflow Utilities for ComfyUI.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .erp_tools import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print(f"[ComfyUI-Studio-Nodes] Failed to load erp_tools: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
