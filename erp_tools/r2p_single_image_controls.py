"""Two visible controls for the separate SeedVR2 still-image workflow."""

from comfy_execution.utils import get_executing_context


class R2PStillUpscaleControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "seedvr2_ema_3b_fp16.safetensors",
                        "seedvr2_ema_7b_fp16.safetensors",
                        "seedvr2_ema_7b_sharp_fp16.safetensors",
                    ],
                ),
                "long_edge": ("INT", {"default": 3200, "min": 512, "max": 8000, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "SEEDVR2_DIT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "dit", "short_edge", "max_edge", "long_edge")
    FUNCTION = "configure"
    CATEGORY = "R2P/Single Image"

    def configure(self, image, model, long_edge):
        height, width = image.shape[1:3]
        short_edge = max(16, round(long_edge * min(width, height) / max(width, height) / 2) * 2)
        dit = {
            "model": model,
            "device": "cuda:0",
            "offload_device": "cpu",
            "cache_model": False,
            "blocks_to_swap": 32,
            "swap_io_components": True,
            "attention_mode": "sdpa",
            "torch_compile_args": None,
            "node_id": get_executing_context().node_id,
        }
        return image, dit, short_edge, long_edge, long_edge


NODE_CLASS_MAPPINGS = {"R2PStillUpscaleControls": R2PStillUpscaleControls}
NODE_DISPLAY_NAME_MAPPINGS = {"R2PStillUpscaleControls": "单张放大 · 模型 / 长边尺寸"}
