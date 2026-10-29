"""Comfy adapters for original-detail panorama finishing; no model or GPU work."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import folder_paths
import comfy.model_management

from .fidelity_core import process_panorama, resize_equirectangular
from .quality_preview import _view, _font


def image_digest(image):
    digest = hashlib.sha256()
    for row in image:
        digest.update(np.ascontiguousarray(row).tobytes())
    return digest.hexdigest()


def inspection_image(baseline, final, mask):
    side, label, gap = 320, 30, 8
    views = [("前方", 0, 0), ("右方", 90, 0), ("闭合处", 180, 0),
             ("左方", -90, 0), ("顶面", 0, 90), ("地面", 0, -90)]
    pair_width = 2 * side + gap
    canvas = Image.new("RGB", (2 * pair_width + 3 * gap, 66 + 3 * (side + label + gap)), "#182028")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), "同一球面视角｜每组左：原图基准；右：结果", fill="white", font=_font(21))
    draw.text((gap, 38), "此图仅供检查，不能作为全景上传。回退区域另存黑白蒙版：白色=保留原图。", fill="#ffd78a", font=_font(16))
    for i, (name, yaw, pitch) in enumerate(views):
        x = gap + (i % 2) * (pair_width + gap)
        y = 66 + (i // 2) * (side + label + gap)
        draw.text((x, y + 4), name + " / 原图                         结果", fill="white", font=_font(16))
        for k, frame in enumerate((baseline, final)):
            view = _view(frame, yaw, pitch, side)
            pixels = np.rint(view.clip(0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels), (x + k * (side + gap), y + label))
    return torch.from_numpy(np.array(canvas)).float().div_(255)


class ERPSourceMasterFinishv4:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",), "guide": ("IMAGE",),
            "output_width": (["4500", "5000", "6000", "7000", "8000", "10000", "12000", "14000", "16000", "20000", "36000"], {"default": "5000"}),
            "strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "只调整受限明暗；0保持原图。AI物体、纹理不进入成品。"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("final", "baseline", "fallback_mask", "inspection", "report")
    FUNCTION = "finish"
    CATEGORY = "panorama/source-fidelity"
    DESCRIPTION = "以原始全景保留形状与表面状态；AI仅提供亮度分布参考。尺寸增大不代表增加真实细节。"

    def finish(self, source, guide, output_width="5000", strength=0.35):
        if source.ndim != 4 or guide.ndim != 4 or source.shape[-1] != 3 or guide.shape[-1] != 3:
            raise ValueError("请连接完整 RGB 全景图，不能连接蒙版或查看器截图。")
        if guide.shape[0] not in (1, source.shape[0]):
            raise ValueError("参考图数量应为1或与原图数量相同，避免批次错配。")
        results, baselines, masks, previews, reports = [], [], [], [], []
        for index, source_frame in enumerate(source):
            comfy.model_management.throw_exception_if_processing_interrupted()
            original = source_frame.detach().cpu().float().numpy()
            reference = guide[min(index, guide.shape[0] - 1)].detach().cpu().float().numpy()
            result, mask, report = process_panorama(original, reference, output_width=int(output_width), strength=strength)
            baseline = resize_equirectangular(original, int(output_width))
            report.update({"source_float32_sha256": image_digest(original),
                           "guide_float32_sha256": image_digest(reference),
                           "final_float32_sha256": image_digest(result),
                           "node": "ERPSourceMasterFinishv4", "batch_index": index,
                           "visual_acceptance": "Requires same-view inspection; automatic checks are not exhaustive.",
                           "fallback_mask_legend": "0=accepted tone; 1=source restored; gray=transition"})
            previews.append(inspection_image(baseline, result, mask))
            results.append(torch.from_numpy(np.ascontiguousarray(result)))
            baselines.append(torch.from_numpy(np.ascontiguousarray(baseline)))
            masks.append(torch.from_numpy(np.ascontiguousarray(mask)))
            reports.append(report)
        return (torch.stack(results), torch.stack(baselines), torch.stack(masks),
                torch.stack(previews), json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))


class ERPSourceMasterAuditSavev4:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "Panorama_v4/audit"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "panorama/source-fidelity"

    def save(self, report, filename_prefix="Panorama_v4/audit", prompt=None, extra_pnginfo=None):
        audits = json.loads(report)
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        # The standard resolver enforces output-directory containment.
        path = Path(folder) / f"{name}_{counter:05}_.json"
        while path.exists():
            counter += 1
            path = Path(folder) / f"{name}_{counter:05}_.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"audits": audits, "prompt": prompt, "workflow": (extra_pnginfo or {}).get("workflow")}
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return {"ui": {"text": [f"已保存检查记录：{subfolder}/{path.name}", report]}}


NODE_CLASS_MAPPINGS = {"ERPSourceMasterFinishv4": ERPSourceMasterFinishv4,
                       "ERPSourceMasterAuditSavev4": ERPSourceMasterAuditSavev4}
NODE_DISPLAY_NAME_MAPPINGS = {"ERPSourceMasterFinishv4": "全景保形清透 · 原图细节优先 v4",
                              "ERPSourceMasterAuditSavev4": "保存全景回退与检查记录 v4"}
