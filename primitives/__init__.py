"""
Flux2 and Essential Primitives Compatibility for ComfyUI
"""
import torch

class EmptyFlux2LatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/flux2"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], dtype=torch.float16)
        return ({"samples": latent},)


class Flux2Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, width, height):
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        return (sigmas,)


class PrimitiveStringMultiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "utils"

    def get_value(self, string=""):
        return (string,)


class PreviewAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "source": ("*", {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def preview(self, source=None):
        text = str(source) if source is not None else ""
        return {"ui": {"text": [text]}, "result": (text,)}


class ReferenceLatentCompat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "latent": ("LATENT",),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning"

    def execute(self, conditioning, latent=None):
        if latent is not None:
            samples = latent.get("samples") if isinstance(latent, dict) else latent
            out = []
            for cond, meta in conditioning:
                new_meta = meta.copy()
                refs = list(new_meta.get("reference_latents", []))
                refs.append(samples)
                new_meta["reference_latents"] = refs
                out.append([cond, new_meta])
            return (out,)
        return (conditioning,)


class ColorTransferCompat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "source": ("IMAGE",),
                "mode": (["mean", "lab", "pdf"], {"default": "mean"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer"
    CATEGORY = "image/color"

    def transfer(self, target, source, mode="mean"):
        # Mean/std color transfer in RGB
        t = target.clone()
        s = source.clone()
        if t.shape[-1] == 3 and s.shape[-1] == 3:
            for b in range(min(t.shape[0], s.shape[0])):
                for c in range(3):
                    t_mean, t_std = t[b, ..., c].mean(), t[b, ..., c].std() + 1e-6
                    s_mean, s_std = s[b, ..., c].mean(), s[b, ..., c].std() + 1e-6
                    t[b, ..., c] = ((t[b, ..., c] - t_mean) / t_std) * s_std + s_mean
        return (torch.clamp(t, 0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "EmptyFlux2LatentImage": EmptyFlux2LatentImage,
    "Flux2Scheduler": Flux2Scheduler,
    "PrimitiveStringMultiline": PrimitiveStringMultiline,
    "PreviewAny": PreviewAny,
    "ReferenceLatent": ReferenceLatentCompat,
    "ColorTransfer": ColorTransferCompat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyFlux2LatentImage": "Empty Flux2 Latent Image",
    "Flux2Scheduler": "Flux2 Scheduler",
    "PrimitiveStringMultiline": "Primitive String (Multiline)",
    "PreviewAny": "Preview Any",
    "ReferenceLatent": "Set Reference Latent",
    "ColorTransfer": "Color Transfer",
}
