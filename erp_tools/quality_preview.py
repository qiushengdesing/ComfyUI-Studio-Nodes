"""Read-only 360 review node; no AI repair, global patches, or new dependencies."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def _view(image, yaw, pitch, side=384, fov=85.0):
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    factor = math.tan(math.radians(fov / 2)) / (side / 2)
    x = (xx - (side - 1) / 2) * factor
    y = -(yy - (side - 1) / 2) * factor
    pitch = math.radians(pitch)
    vertical = y * math.cos(pitch) + math.sin(pitch)
    forward = -y * math.sin(pitch) + math.cos(pitch)
    longitude = np.arctan2(x, forward) + math.radians(yaw)
    latitude = np.arctan2(vertical, np.sqrt(x * x + forward * forward))
    mx = (((longitude / (2 * np.pi) + .5) * width - .5) % width).astype(np.float32)
    my = np.clip((.5 - latitude / np.pi) * height - .5, 0, height - 1).astype(np.float32)
    return cv2.remap(image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def _font(size):
    # Loading an installed font is read-only; fallback labels remain in the report.
    for name in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def _as_rgb(image, name):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"{name}需要 BHWC RGB 图像。")
    if image.shape[0] != 1:
        raise ValueError(f"{name}需要单张完整全景；请逐张检查，避免错配原图。")
    h, w = image.shape[1:3]
    if w != 2 * h:
        raise ValueError(f"{name}必须为精确2:1全景，实际为{w}×{h}。请连接完整视野准备节点或正确还原后的全景。")
    if h < 32 or not image.is_floating_point() or not torch.isfinite(image).all().item():
        raise ValueError(f"{name}包含无效尺寸或像素。")
    return (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)


class ERPReview360v2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference": ("IMAGE",), "result": ("IMAGE",)},
                "optional": {"preview_size": (["384", "512"],)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diagnostic_only", "review_report")
    FUNCTION = "review"
    CATEGORY = "panorama/validated-sizing"

    def review(self, reference, result, preview_size="384"):
        if str(preview_size) not in ("384", "512"):
            raise ValueError("检查图尺寸请选择384或512。")
        before = _as_rgb(reference, "原图参考")
        after = _as_rgb(result, "输出结果")
        side = int(preview_size)
        views = [("接缝·平视", 180, 0), ("接缝·顶面", 180, 45),
                 ("接缝·地面", 180, -45), ("相对面", 0, 0),
                 ("顶部极点", 180, 90), ("地面极点", 180, -90)]
        top = 54
        label_h = 34
        gap = 8
        pair_w = 2 * side + gap
        row_h = side + label_h
        canvas = Image.new("RGB", (2 * pair_w + 3 * gap, top + 3 * (row_h + gap)), "#161b20")
        draw = ImageDraw.Draw(canvas)
        heading = _font(20)
        labels = _font(16)
        draw.text((gap, 6), "360°检查图｜每组左：原图；右：处理后｜仅供检查，不能当全景上传", fill="#ffe3a4", font=heading)
        draw.text((gap, 30), "请检查断缝、重复物体、物体增删、材料变化、顶地变形；尺寸正确不等于视觉通过。", fill="#ffffff", font=labels)
        for index, (name, yaw, pitch) in enumerate(views):
            col, row = index % 2, index // 2
            x, y = gap + col * (pair_w + gap), top + row * (row_h + gap)
            draw.text((x + 3, y + 5), name + "  原图", fill="#d0e2ef", font=labels)
            draw.text((x + side + gap + 3, y + 5), "处理后", fill="#d0e2ef", font=labels)
            a = _view(before, yaw, pitch, side)
            b = _view(after, yaw, pitch, side)
            canvas.paste(Image.fromarray(a), (x, y + label_h))
            canvas.paste(Image.fromarray(b), (x + side + gap, y + label_h))
        output = torch.from_numpy(np.asarray(canvas).copy()).float().div(255).unsqueeze(0)
        report = (
            f"原图参考 {before.shape[1]}×{before.shape[0]}；输出 {after.shape[1]}×{after.shape[0]}，比例均为2:1。"
            "已生成六方向的原图/结果对照：接缝平视、接缝顶面、接缝地面、相对面、顶部极点和地面极点。"
            "【视觉状态：待核验】本节点只重投影供检查，不修复图像，不自动判断物体保持，也不会将尺寸检查称为视觉合格。"
            "每组左边为原图，右边为处理后；诊断拼图不是全景成品，不要上传全景平台。"
            "先确认接缝附近没有重复物体和断线，再检查其余方向。六个视角不是整个球面所有细节的穷尽检验。")
        return output, report


NODE_CLASS_MAPPINGS = {"ERPReview360v2": ERPReview360v2}
NODE_DISPLAY_NAME_MAPPINGS = {"ERPReview360v2": "360全景 · 六方向原图结果检查 v2"}
