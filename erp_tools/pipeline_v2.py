"""Strict whole-sphere sizing for the local SeedVR2 panorama workflow.

These nodes validate the size chain, not semantic or geometric fidelity.
They do not patch models, change global devices/threads, or crop ERP content.
"""
import math

import torch
import torch.nn.functional as F


CATEGORY = "panorama/validated-sizing"
MAP_VERSION = "erp-seedvr-side-v2"


def _check_image(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("需要 BHWC 格式的 RGB 图像；请连接加载图像节点。")
    if not image.is_floating_point() or not torch.isfinite(image).all().item():
        raise ValueError("图像必须是有限数值的浮点 RGB 数据。")
    if image.shape[0] < 1 or image.shape[1] < 32 or image.shape[2] < 64:
        raise ValueError("全景图尺寸过小或批次为空。")
    return int(image.shape[2]), int(image.shape[1])


def _resize_full(image, width, height):
    if tuple(image.shape[1:3]) == (height, width):
        return image.clone()
    return F.interpolate(image.movedim(-1, 1).float(), size=(height, width),
                         mode="bicubic", align_corners=False, antialias=True
                         ).movedim(1, -1).clamp(0, 1).to(image.dtype)


class ERPPrepare360v2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": (["2048", "1536", "1024"],),
            "input_kind": (["完整360全景"],),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, image, width="2048", input_kind="完整360全景"):
        source_w, source_h = _check_image(image)
        if image.shape[0] != 1:
            raise ValueError('每次只接受一张静态完整全景；动态图请先选择单帧。已在生成前停止。')
        if input_kind != "完整360全景":
            raise ValueError("此流程只接受覆盖完整 360×180 度的等距柱状全景原图。")
        if str(width) not in ("1024", "1536", "2048"):
            raise ValueError("洗图宽度请选择 1024、1536 或 2048。")
        deviation = abs(source_w / (2.0 * source_h) - 1.0)
        if deviation > 0.01 + 1e-12:
            raise ValueError(
                f"输入为 {source_w}×{source_h}，与完整 2:1 全景相差 {deviation:.2%}，超过 1%。"
                "已停止，避免裁掉视野或拼出重复区域。请使用完整 360×180 度全景原文件，"
                "不要使用查看器截图、被裁过的图，或上一版的 1024×528 输出。")
        target_w = int(width)
        target_h = target_w // 2
        result = _resize_full(image, target_w, target_h)
        report = (
            f"输入 {source_w}×{source_h} → 整图缩放为 {target_w}×{target_h}，精确 2:1；"
            f"输入比例偏差 {deviation:.3%}。保留整张图的覆盖范围，无裁切、无黑边。"
            "前提是输入确为完整 360×180 度全景；尺寸检查不能证明投影正确，"
            "也不保证后续生成的物体形状不变。")
        return result, target_w, target_h, report


class ERPWrapPadv2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "padding_fraction": ("FLOAT", {"default": 0.25, "min": 0.0625,
                                             "max": 0.5, "step": 0.0625}),
        }}

    RETURN_TYPES = ("IMAGE", "ERPMAP")
    RETURN_NAMES = ("padded_image", "wrap_map")
    FUNCTION = "pad"
    CATEGORY = CATEGORY

    def pad(self, image, padding_fraction=0.25):
        width, height = _check_image(image)
        if width != 2 * height:
            raise ValueError(f"环绕放大输入必须精确 2:1，实际为 {width}×{height}。"
                             "请从新版全景准备节点重新洗图；不要裁切旧结果来凑比例。")
        fraction = float(padding_fraction)
        if not math.isfinite(fraction) or not 0.0625 <= fraction <= 0.5:
            raise ValueError("环绕扩边比例必须在 0.0625 至 0.5 之间。")
        pad = max(1, int(math.floor(width * fraction + 0.5)))
        indices = torch.arange(-pad, width + pad, device=image.device).remainder(width)
        padded = torch.index_select(image, 2, indices)
        mapping = {"version": MAP_VERSION, "width": width, "height": height,
                   "left": pad, "right": pad, "padded_width": width + 2 * pad,
                   "padded_height": height, "batch": int(image.shape[0]),
                   "upscaler": "SeedVR2VideoUpscaler/side/max_resolution=0"}
        return padded, mapping


def _validate_map(mapping):
    if not isinstance(mapping, dict) or mapping.get("version") != MAP_VERSION:
        raise ValueError("缺少新版环绕记录，请将同一个 ERPWrapPadv2 的 wrap_map 接到还原节点。")
    keys = ("width", "height", "left", "right", "padded_width", "padded_height", "batch")
    if any(type(mapping.get(key)) is not int or mapping[key] <= 0 for key in keys):
        raise ValueError("环绕记录包含无效尺寸。")
    width, height = mapping["width"], mapping["height"]
    if (width != height * 2 or mapping["padded_width"] != width + mapping["left"] + mapping["right"]
            or mapping["padded_height"] != height or mapping["left"] != mapping["right"]
            or mapping["left"] > width // 2
            or mapping.get("upscaler") != "SeedVR2VideoUpscaler/side/max_resolution=0"):
        raise ValueError("环绕记录不一致，禁止猜测裁切范围。")


class ERPWrapRestorev2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "wrap_map": ("ERPMAP",),
            "final_width": ("INT", {"default": 4500, "min": 256, "max": 65536, "step": 4}),
        }, "optional": {"blend_overlap": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panorama", "report")
    FUNCTION = "restore"
    CATEGORY = CATEGORY

    def restore(self, image, wrap_map, final_width=4500, blend_overlap=True):
        actual_w, actual_h = _check_image(image)
        _validate_map(wrap_map)
        if type(final_width) is not int or not 256 <= final_width <= 65536 or final_width % 4:
            raise ValueError("最终宽度必须是 256 至 65536 之间的 4 的倍数，例如 4500、6000、8000、36000。")
        target_h = final_width // 2
        if image.shape[0] != wrap_map["batch"]:
            raise ValueError("放大结果批次与环绕记录不一致。")

        # Match the installed SeedVR2: torchvision short-side resize truncates W;
        # setup_video_transform floors true W/H to even, then slices from top-left.
        # Use the PRE-even width for the coordinate transform: the final one-pixel
        # right trim is not an isotropic resize and must not alter this scale.
        raw_w = (wrap_map["padded_width"] * target_h) // wrap_map["height"]
        expected_w = (raw_w // 2) * 2
        if (actual_w, actual_h) != (expected_w, target_h):
            raise ValueError(
                f"环绕放大尺寸不匹配：实际 {actual_w}×{actual_h}，"
                f"此记录应为 {expected_w}×{target_h}。已停止，避免错误裁回 360 度。"
                f"请将 SeedVR 的 resolution 设为 {target_h}、max_resolution 设为 0，"
                "并直接连接其输出；中间不要插入裁切、补边或缩放。")

        scale_x = raw_w / wrap_map["padded_width"]
        x0 = wrap_map["left"] * scale_x
        span = wrap_map["width"] * scale_x
        # When the interval already maps to whole output pixels, copy them
        # directly. Exact comparisons are intentional: never round a fractional
        # mapping into a crop, or alter legacy blend=True interpolation behavior.
        exact_crop = not blend_overlap and span == final_width and x0.is_integer()
        if exact_crop:
            ix = int(x0)
            result = image[:, :, ix:ix + final_width].clone()
        else:
            # Pixel-center mapping of the original whole ERP interval,
            # resampled once. Fractional x0/span must stay fractional.
            x = x0 + (torch.arange(final_width, device=image.device, dtype=torch.float32) + 0.5) * (span / final_width)
            y = torch.arange(target_h, device=image.device, dtype=torch.float32) + 0.5
            gx = 2.0 * x / actual_w - 1.0
            gy = 2.0 * y / actual_h - 1.0
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
            result = F.grid_sample(image.movedim(-1, 1).float(), grid, mode="bilinear",
                                   padding_mode="border", align_corners=False).movedim(1, -1)
        # SeedVR can predict slightly different pixels for the two copies of
        # the same longitude. Fold only a narrow overlap onto the exact same
        # longitude, with symmetric weights at closure; never retain extra FoV.
        if blend_overlap:
            band = max(1, min(round(final_width * .02),
                              int(wrap_map['left'] / wrap_map['width'] * final_width)))
            for start, stop, offset in ((0, band, span), (final_width-band, final_width, -span)):
                xx = x[start:stop] + offset
                gy2, gx2 = torch.meshgrid(gy, 2.0 * xx / actual_w - 1.0, indexing='ij')
                overlap_grid = torch.stack((gx2,gy2),dim=-1).unsqueeze(0).expand(image.shape[0],-1,-1,-1)
                overlap = F.grid_sample(image.movedim(-1,1).float(),overlap_grid,
                                        mode='bilinear',padding_mode='border',align_corners=False).movedim(1,-1)
                distance = torch.arange(start,stop,device=image.device,dtype=torch.float32)+.5
                if start:distance=final_width-distance
                t=(distance/band).clamp(0,1)
                weight=(.5*(1-t*t*(3-2*t)))[None,None,:,None]
                result[:,:,start:stop]=torch.lerp(result[:,:,start:stop],overlap,weight)
        if not exact_crop:
            result = result.clamp(0, 1).to(image.dtype)
        overlap_report = (
            "首尾各2%使用同经度重复上下文融合，可能轻微软化细线。" if blend_overlap else
            "重复上下文融合已关闭：仅采样中央完整一圈，不将左右扩边的另一份生成内容混入。")
        report = (
            f"尺寸链检查通过：原全景 {wrap_map['width']}×{wrap_map['height']}，"
            f"左右各扩 {wrap_map['left']}，SeedVR 实际 {actual_w}×{actual_h}；"
            f"按原完整一圈对应的浮点区间 [{x0:.6f}, {x0 + span:.6f}) 还原，"
            f"输出 {final_width}×{target_h}，精确 2:1。"
            "已处理 SeedVR 右侧取偶裁边；完整纵向范围保留。" + overlap_report +
            "此检查只保证本尺寸链，不能识别同尺寸的内容替换，也不代表接缝、物体和几何已通过视觉验收。")
        return result, report


NODE_CLASS_MAPPINGS = {"ERPPrepare360v2": ERPPrepare360v2,
                       "ERPWrapPadv2": ERPWrapPadv2,
                       "ERPWrapRestorev2": ERPWrapRestorev2}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ERPPrepare360v2": "360全景 · 完整视野准备 v2",
    "ERPWrapPadv2": "360全景 · 环绕扩边与记录 v2",
    "ERPWrapRestorev2": "360全景 · 校验并还原整圈 v2",
}
