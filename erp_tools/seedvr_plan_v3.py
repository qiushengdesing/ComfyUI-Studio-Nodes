"""Choose smaller wrap canvases and full block swapping for large panoramas.

Only a COPY of SeedVR's configuration dictionary is changed. No weights are
loaded, no devices are probed, and no image/VRAM threshold is introduced here.
"""
import re


WIDTHS = (4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 20000, 36000)


def _device_key(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value in ("cpu", "cpu:0"):
        return "cpu"
    if value == "cuda":
        return "cuda:0"
    if re.fullmatch(r"cuda:\d+", value):
        return value
    return None


class ERPSeedVRPlanv3:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dit": ("SEEDVR2_DIT",),
            "final_width": ("INT", {"forceInput": True, "default": 4500}),
        }}

    RETURN_TYPES = ("SEEDVR2_DIT", "FLOAT", "STRING")
    RETURN_NAMES = ("dit", "padding_fraction", "report")
    FUNCTION = "plan"
    CATEGORY = "panorama/validated-sizing"

    def plan(self, dit, final_width=4500):
        if not isinstance(dit, dict):
            raise ValueError("请连接 SeedVR2 DiT 加载器输出的配置。")
        if type(final_width) is not int or final_width not in WIDTHS:
            raise ValueError(f"最终宽度仅支持: {', '.join(str(w) for w in WIDTHS)}。")
        # Only top-level entries are changed. Nested compile settings and any
        # loader metadata retain their exact values and identities.
        result = dict(dit)
        height = final_width // 2
        if final_width < 7000:
            return result, 0.25, (
                f"最终 {final_width}×{height}：保留所选 DiT 配置，左右各补原宽度25%。"
                f"放大中间图为 {final_width * 3 // 2}×{height}；"
                "按环绕记录还原完整一圈，尺寸选择不改变所选放大模型。")

        model = dit.get("model")
        basename = model.replace("\\", "/").rsplit("/", 1)[-1].lower() if isinstance(model, str) else ""
        family = re.fullmatch(r"seedvr2_ema_(3b|7b)[_-].+\.(safetensors|gguf)", basename)
        if family is None:
            raise ValueError(
                f"{final_width}大图自动配置仅识别 SeedVR2 EMA 3B/7B 标准命名模型；"
                f"当前模型 {model!r} 无法确认层数，已停止，不能猜测 blocks_to_swap。")
        device = _device_key(dit.get("device"))
        if device is None:
            raise ValueError(f"{final_width}大图自动配置仅支持加载器的 CPU/CUDA 推理设备，请检查 device。")

        if device == "cpu":
            result["blocks_to_swap"] = 0
            result["swap_io_components"] = False
            transfer_report = "CPU推理，不启用层交换或输入输出组件交换。"
        else:
            result["blocks_to_swap"] = 36 if family.group(1) == "7b" else 32
            result["swap_io_components"] = True
            offload = _device_key(dit.get("offload_device"))
            if offload is None or offload == device:
                result["offload_device"] = "cpu"
            else:
                result["offload_device"] = offload
            # A different valid loader-selected CUDA device remains respected;
            # actual hardware availability is validated by the loader/runtime.
            transfer_report = (
                f"{family.group(1).upper()} 使用 {result['blocks_to_swap']} 层交换，"
                f"输入输出组件同时交换到 {result['offload_device']}。")
        padding_fraction = 0.125 if final_width == 7000 else 0.0625
        context = int(final_width * padding_fraction)
        padded_width = final_width + 2 * context
        fusion_band = round(final_width * .02)
        reduction = (1 - (1 + 2 * padding_fraction) / 1.5) * 100
        report = (
            f"最终{final_width}×{height}：左右各补原宽度{padding_fraction * 100:g}%，"
            f"中间图{padded_width}×{height}；根据环绕记录还原完整一圈，"
            f"每侧{context}像素上下文覆盖{fusion_band}像素接缝融合带。"
            + transfer_report +
            "仅复制并调整本次DiT配置，不修改权重、上游配置、缓存/注意力/编译设置。"
            f"相比25%补边，中间像素减少{reduction:.2f}%，实际耗时与显存不保证同比下降；"
            "接缝及邻近物体仍需实图检查。")
        return result, padding_fraction, report


NODE_CLASS_MAPPINGS = {"ERPSeedVRPlanv3": ERPSeedVRPlanv3}
NODE_DISPLAY_NAME_MAPPINGS = {"ERPSeedVRPlanv3": "全景放大 · 大图自动配置 v3"}
