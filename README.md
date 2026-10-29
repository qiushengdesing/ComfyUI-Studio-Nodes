# ComfyUI-Studio-Nodes (ERP Seam & Polar Guard Tools)

Dedicated 360° Equirectangular Panorama (ERP) seam healing, Shaniu 6.0 algorithm, and polar structure protection nodes for ComfyUI.

## Features

- **Shaniu 6.0 Native Seam Healing (`Shaniu6NativeSeamHeal`)**:
  - Longitudinal 1D Luma Profile + Segment Contrast Normalization + Viterbi Dynamic Programming geometric alignment.
  - Dual-band harmonic tone equalization (broad low-frequency + row-level mid-frequency).
  - Circular interlaced 50% boundary fusion with exact zero seam error.
  - Polar 3D spherical singularity closure (eliminates polar pinching and vortex distortion).
- **ERP Boundary & Wrap Protection (`ERPWrapPadv2`, `ERPWrapRestorev2`, `ERPPrepare360v2`)**:
  - Seamless horizontal wrap padding and restoration preserving original coordinate bounds.
  - Complete 360° multi-view preparation.
- **Seam Tone & Color Repair (`ERPSeamToneRepair`)**:
  - Automatic boundary seam tone harmonization and smoothing.
- **Polar Structure Protection (`ERPPoleSourceGuardv2`, `ERPPoleSourceGuardv3`)**:
  - Ceiling & ground zenith/nadir distortion guards.
- **Review & Inspection (`ERPReview360v2`)**:
  - Auto-unfold 6 cubemap/directional views (original vs result).

## Pure & Non-Conflicting Architecture

This repository strictly contains only the author's original `erp_tools` suite.
- **Zero Core Node Overwrites**: Does not register or overwrite any ComfyUI core nodes (`EmptyFlux2LatentImage`, `Flux2Scheduler`, etc.).
- **Zero Third-Party Repackaging**: Pure standalone implementation. Third-party upscalers or viewers should be installed directly from their official repositories.

## Installation

Inside your ComfyUI root directory:

```bash
cd custom_nodes
git clone https://github.com/qiushengdesing/ComfyUI-Studio-Nodes.git
```

Install requirements:

```bash
pip install -r ComfyUI-Studio-Nodes/requirements.txt
```

Restart ComfyUI.

## Recommended Companion Plugins

If your workflow requires 360 viewer preview, SeedVR2 upscaler, or image comparison, install the respective official community packages:
- **360 Viewer & DiT360**: [ComfyUI-DiT360Plus](https://github.com/thomashollier/ComfyUI-DiT360Plus)
- **SeedVR2 Video/Image Upscaler**: [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/ainvfx/ComfyUI-SeedVR2_VideoUpscaler)
- **Image Comparison**: [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)

## License
MIT
