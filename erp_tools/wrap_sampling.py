"""Experimental panorama sampling nodes for Flux2 Klein 9B; not installed.

Use a native ComfyUI sampler. The installed Flux2KleinKSamplerExperimental
bypasses ModelPatcher wrapper preparation and is not supported here.
Halos add neighboring context; they do not make Flux RoPE truly periodic.
"""

import torch

from comfy.patcher_extension import WrappersMP


WRAPPER_KEY = "panorama_flux2_horizontal_halo_prototype"
DUAL_VIEW_KEY = "panorama_flux2_dual_view_prototype"
CATEGORY = "panorama/experimental"


def _wrap_width(value, halo, dim):
    width = value.shape[dim]
    if not 0 <= halo <= width:
        raise ValueError(f"Horizontal halo {halo} must be between 0 and width {width}.")
    if halo == 0:
        return value
    return torch.cat((value.narrow(dim, width - halo, halo), value,
                      value.narrow(dim, 0, halo)), dim=dim)


def _validate_diffusion_inputs(x, ref_latents, control, transformer_options, kwargs):
    if x.ndim != 4 or x.shape[1] != 128:
        raise ValueError("Panorama sampling requires Flux2 BCHW latents with 128 channels; concat/control inputs are unsupported.")
    if control is not None:
        raise ValueError("Panorama sampling prototypes do not support ControlNet residuals.")
    options = {} if transformer_options is None else transformer_options.copy()
    for name in ("patches", "patches_replace", "rope_options"):
        if options.get(name):
            raise ValueError(f"Panorama sampling prototypes do not support transformer_options[{name!r}].")
    for name, value in kwargs.items():
        if name == "ref_latents_method" and value in (None, "index"):
            continue
        if value is not None:
            raise ValueError(f"Panorama sampling prototypes do not support nonempty {name!r}; spatial conditions require explicit handling.")
    if ref_latents is not None:
        for ref in ref_latents:
            if ref.ndim != 4 or ref.shape[1:] != x.shape[1:]:
                raise ValueError("Every reference must be a full ERP latent with the same channels, height and width as the target.")
            if ref.shape[0] not in (1, x.shape[0]):
                raise ValueError("Reference batch must be 1 or match the target batch.")
    return options


def _check_model(model):
    config = model.model.model_config.unet_config
    expected = {"image_model": "flux2", "in_channels": 128,
                "out_channels": 128, "patch_size": 1, "hidden_size": 4096}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("This prototype targets the local Flux2 Klein 9B architecture only (128 channels, patch 1, hidden size 4096).")


class Flux2HorizontalHalo:
    def __init__(self, halo_latents):
        self.halo_latents = int(halo_latents)

    def __call__(self, executor, x, timestep, context, y=None, guidance=None,
                 ref_latents=None, control=None, transformer_options=None, **kwargs):
        options = _validate_diffusion_inputs(x, ref_latents, control, transformer_options, kwargs)

        refs = None
        if ref_latents is not None:
            refs = []
            for ref in ref_latents:
                refs.append(_wrap_width(ref, self.halo_latents, -1))

        # Repeat the CURRENT noisy state at each evaluation, never fresh halo noise.
        padded = _wrap_width(x, self.halo_latents, -1)
        prediction = executor(padded, timestep, context, y, guidance, refs,
                              control, options, **kwargs)
        if prediction.shape != padded.shape:
            raise ValueError(f"Unexpected denoiser shape {tuple(prediction.shape)}; expected {tuple(padded.shape)}.")
        return prediction[..., self.halo_latents:self.halo_latents + x.shape[-1]].clone()


class PanoramaFlux2WrapModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "halo_latents": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, halo_latents=8):
        _check_model(model)
        cloned = model.clone()
        cloned.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        if halo_latents:
            embedded = cloned.model_options.get("transformer_options", {}).get("wrappers", {}).get(WrappersMP.DIFFUSION_MODEL, {})
            if cloned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) or any(embedded.values()):
                raise ValueError("Use an unwrapped MODEL for this prototype; other diffusion-model wrappers need a compatibility check.")
            cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                                        Flux2HorizontalHalo(halo_latents))
        return (cloned,)


class Flux2DualView:
    def __call__(self, executor, x, timestep, context, y=None, guidance=None,
                 ref_latents=None, control=None, transformer_options=None, **kwargs):
        options = _validate_diffusion_inputs(x, ref_latents, control, transformer_options, kwargs)
        width = x.shape[-1]
        shift = width // 2
        prediction = executor(x, timestep, context, y, guidance, ref_latents,
                              control, options.copy(), **kwargs)
        rolled_refs = None if ref_latents is None else [ref.roll(shift, -1) for ref in ref_latents]
        rolled_prediction = executor(x.roll(shift, -1), timestep, context, y,
                                     guidance, rolled_refs, control, options.copy(), **kwargs)
        if prediction.shape != x.shape or rolled_prediction.shape != x.shape:
            raise ValueError("Dual-view denoiser predictions must match the original latent shape.")
        rolled_prediction = rolled_prediction.roll(-shift, -1)
        columns = torch.arange(width, device=prediction.device, dtype=torch.float32)
        weight = torch.sin(torch.pi * (columns + 0.5) / width).square().view(1, 1, 1, width)
        # Blend flow predictions before the sampler update, retaining the native output dtype.
        return torch.lerp(rolled_prediction.float(), prediction.float(), weight).to(prediction.dtype)


class PanoramaFlux2DualViewModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model):
        _check_model(model)
        cloned = model.clone()
        cloned.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, DUAL_VIEW_KEY)
        embedded = cloned.model_options.get("transformer_options", {}).get("wrappers", {}).get(WrappersMP.DIFFUSION_MODEL, {})
        if cloned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) or any(embedded.values()):
            raise ValueError("Use an unwrapped MODEL for dual-view sampling; do not stack it with halo or other diffusion-model wrappers.")
        cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, DUAL_VIEW_KEY, Flux2DualView())
        return (cloned,)


def _check_vae(vae):
    if (vae.latent_channels != 128 or vae.spacial_compression_encode() != 16
            or vae.spacial_compression_decode() != 16):
        raise ValueError("Horizontal halo VAE nodes require the Flux2 128-channel, 16x VAE.")


class PanoramaFlux2WrapVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pixels": ("IMAGE",),
            "vae": ("VAE",),
            "halo_pixels": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 16}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, pixels, vae, halo_pixels=128):
        _check_vae(vae)
        if pixels.ndim != 4 or pixels.shape[-1] not in (3, 4):
            raise ValueError("Expected a BHWC RGB or RGBA image.")
        if pixels.shape[1] % 16 or pixels.shape[2] % 16 or halo_pixels % 16:
            raise ValueError("Image height, width and horizontal halo must be multiples of 16 to preserve Flux2 latent alignment.")
        padded = _wrap_width(pixels[..., :3], halo_pixels, 2)
        encoded = vae.encode(padded)
        expected = (pixels.shape[0], 128, pixels.shape[1] // 16,
                    (pixels.shape[2] + 2 * halo_pixels) // 16)
        if tuple(encoded.shape) != expected:
            raise ValueError(f"Unexpected VAE encode shape {tuple(encoded.shape)}; expected {expected}.")
        start = halo_pixels // 16
        samples = encoded[..., start:start + pixels.shape[2] // 16].clone()
        return ({"samples": samples},)


class PanoramaFlux2WrapVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "halo_latents": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, samples, vae, halo_latents=8):
        _check_vae(vae)
        latent = samples["samples"]
        if latent.ndim != 4 or latent.shape[1] != 128:
            raise ValueError("Expected Flux2 BCHW latents with 128 channels.")
        padded = _wrap_width(latent, halo_latents, -1)
        decoded = vae.decode(padded)
        expected = (latent.shape[0], latent.shape[2] * 16,
                    (latent.shape[3] + 2 * halo_latents) * 16, 3)
        if tuple(decoded.shape) != expected:
            raise ValueError(f"Unexpected VAE decode shape {tuple(decoded.shape)}; expected {expected}.")
        start = halo_latents * 16
        return (decoded[:, :, start:start + latent.shape[-1] * 16, :].clone(),)


NODE_CLASS_MAPPINGS = {
    "PanoramaFlux2DualViewModelPrototype": PanoramaFlux2DualViewModel,
    "PanoramaFlux2WrapModelPrototype": PanoramaFlux2WrapModel,
    "PanoramaFlux2WrapVAEEncodePrototype": PanoramaFlux2WrapVAEEncode,
    "PanoramaFlux2WrapVAEDecodePrototype": PanoramaFlux2WrapVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaFlux2DualViewModelPrototype": "Panorama Klein 9B Dual-View Sampling (Prototype)",
    "PanoramaFlux2WrapModelPrototype": "Panorama Klein 9B Horizontal Halo (Prototype)",
    "PanoramaFlux2WrapVAEEncodePrototype": "Panorama Flux2 VAE Encode with Halo (Prototype)",
    "PanoramaFlux2WrapVAEDecodePrototype": "Panorama Flux2 VAE Decode with Halo (Prototype)",
}
