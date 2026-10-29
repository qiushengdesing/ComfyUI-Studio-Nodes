"""CPU-only, original-pixel panorama tone transfer.

The guide contributes a luminance histogram, never spatial detail. RGB values
are display-referred float32 in [0, 1]; this is a conservative photographic tone
adjustment, not restoration of information missing from the source.
"""

import math

import cv2
import numpy as np


OUTPUT_WIDTHS = (4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 20000, 36000)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
STRIP_ROWS = 48
TONE_SLOPE_DEVIATION = 0.04


def _validate_image(image, name):
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError(f"{name} 必须是 float32 NumPy RGB 数组，数值范围 0–1。")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} 必须是 H×W×3 RGB，不能含透明通道或批次维度。")
    h, w = image.shape[:2]
    if h < 2 or w != 2 * h:
        raise ValueError(f"{name} 必须是完整 2:1 全景；收到 {w}×{h}。不能裁掉视野凑比例。")
    for start in range(0, h, STRIP_ROWS):
        block = image[start:start + STRIP_ROWS]
        if not np.isfinite(block).all() or np.min(block) < 0 or np.max(block) > 1:
            raise ValueError(f"{name} 包含非有限值或超出 0–1 的像素。")


def _validate_width(width):
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
        raise ValueError("输出宽度必须是正偶数整数。")
    if width < 4 or width % 2:
        raise ValueError("输出宽度必须是至少 4 的偶数，才能保留精确 2:1。")
    return int(width)


def _smoothstep(value, low, high):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fold_latitude(indices, height):
    folded = np.mod(indices, 2 * height)
    cross_pole = folded >= height
    rows = np.where(cross_pole, 2 * height - 1 - folded, folded)
    return rows, cross_pole


def _resample_rows(image, output_width, start, end):
    """Bilinear pixel-center sampling with periodic longitude and pole folding."""
    height, width = image.shape[:2]
    output_height = output_width // 2
    if output_width == width and 0 <= start <= end <= height:
        return image[start:end].copy()
    x = (np.arange(output_width, dtype=np.float64) + 0.5) * width / output_width - 0.5
    y = (np.arange(start, end, dtype=np.float64) + 0.5) * height / output_height - 0.5
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    fx = (x - xi).astype(np.float32)[None, :, None]
    fy = (y - yi).astype(np.float32)[:, None, None]

    def row_samples(raw_rows):
        rows, crossed = _fold_latitude(raw_rows, height)
        cols = np.mod(xi[None, :] + crossed[:, None] * (width // 2), width)
        left = image[rows[:, None], cols]
        right = image[rows[:, None], (cols + 1) % width]
        return left + (right - left) * fx

    top = row_samples(yi)
    bottom = row_samples(yi + 1)
    return np.ascontiguousarray(top + (bottom - top) * fy, dtype=np.float32)


def resize_equirectangular(source_rgb, output_width):
    """Resize the entire sphere; equal size is an exact copy, with no filtering."""
    _validate_image(source_rgb, "原图")
    output_width = _validate_width(output_width)
    if source_rgb.shape[1] == output_width:
        return source_rgb.copy()
    if output_width < source_rgb.shape[1]:
        # Area footprints cover the complete input without extrapolation or crop.
        return cv2.resize(source_rgb, (output_width, output_width // 2), interpolation=cv2.INTER_AREA)
    result = np.empty((output_width // 2, output_width, 3), dtype=np.float32)
    for start in range(0, len(result), STRIP_ROWS):
        end = min(start + STRIP_ROWS, len(result))
        result[start:end] = _resample_rows(source_rgb, output_width, start, end)
    return result


def _analysis_image(image, width):
    width = min(width, image.shape[1])
    return resize_equirectangular(image, width)


def _luma(image):
    return image[..., 0] * LUMA[0] + image[..., 1] * LUMA[1] + image[..., 2] * LUMA[2]


def build_tone_lut(source_rgb, guide_rgb, strength=0.35):
    """Build a bounded monotone LUT from global brightness quantiles only."""
    _validate_image(source_rgb, "原图分析图")
    _validate_image(guide_rgb, "参考分析图")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("明暗参考强度必须在 0–1。")
    x = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    quantiles = np.array([0.005, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 0.995])
    source_q = np.quantile(_luma(source_rgb), quantiles)
    guide_q = np.quantile(_luma(guide_rgb), quantiles)
    report = {"quantiles": quantiles.tolist(), "source_luminance": source_q.tolist(),
              "guide_luminance": guide_q.tolist(), "strength": float(strength)}
    if strength == 0 or np.array_equal(source_q, guide_q):
        report.update({"identity": True, "gain_limit": 0.04, "slope_min": 1.0, "slope_max": 1.0})
        return x.astype(np.float32), report
    unique, inverse = np.unique(source_q, return_inverse=True)
    residual = np.bincount(inverse, weights=guide_q - source_q) / np.bincount(inverse)
    delta = np.interp(x, unique, residual) * strength
    delta = np.clip(delta, -0.04 * x, 0.04 * x)
    delta *= 1.0 - _smoothstep(x, 0.72, 0.94)
    delta = cv2.GaussianBlur(delta[None, :], (0, 0), sigmaX=12.0)[0]
    delta *= _smoothstep(x, 0.0, 0.025) * (1.0 - _smoothstep(x, 0.90, 0.96))
    delta = np.clip(delta, -0.04 * x, 0.04 * x)
    delta[0] = delta[-1] = 0.0
    max_derivative = float(np.max(np.abs(np.diff(delta)))) * (len(x) - 1)
    if max_derivative > TONE_SLOPE_DEVIATION:
        delta *= TONE_SLOPE_DEVIATION / max_derivative
    lut = (x + delta).astype(np.float32)
    slopes = np.diff(lut.astype(np.float64)) * (len(lut) - 1)
    report.update({"identity": bool(np.array_equal(lut, x.astype(np.float32))),
                   "gain_limit": 0.04, "slope_deviation_limit": TONE_SLOPE_DEVIATION,
                   "slope_min": float(slopes.min()),
                   "slope_max": float(slopes.max()), "maximum_luminance_change": float(np.max(np.abs(delta)))})
    return lut, report


def _features(gray_with_halo):
    center = gray_with_halo[1:-1]
    left, right = np.roll(center, 1, axis=1), np.roll(center, -1, axis=1)
    up, down = gray_with_halo[:-2], gray_with_halo[2:]
    high = center - (left + right + up + down) * 0.25
    edge = np.maximum(np.abs(right - left), np.abs(down - up)) * 0.5
    return high, edge


def _edge_protection_rows(source, output_width, start, end):
    """Keep a narrow, exact source band around resolved lines, not every grain."""
    halo = _resample_rows(source, output_width, start - 8, end + 8)
    _, edge = _features(_luma(halo))
    marked = (edge >= 0.015).astype(np.float32)
    padded = np.pad(marked, ((0, 0), (7, 7)), mode="wrap")
    exact_band = cv2.dilate(padded, np.ones((5, 5), np.uint8))
    transition = cv2.GaussianBlur(exact_band, (9, 9), 1.8)
    protect = np.maximum(exact_band, transition)
    return protect[7:7 + end - start, 7:-7]


def _pad_spherical_map(values, pad):
    h, w = values.shape
    rows, crossed = _fold_latitude(np.arange(-pad, h + pad), h)
    extended = values[rows].copy()
    for index in np.flatnonzero(crossed):
        extended[index] = np.roll(extended[index], w // 2)
    return np.pad(extended, ((0, 0), (pad, pad)), mode="wrap")


def _smooth_protection(protect):
    padded = _pad_spherical_map(protect, 5)
    expanded = cv2.dilate(padded, np.ones((3, 3), np.uint8))
    smoothed = cv2.GaussianBlur(expanded, (9, 9), 1.5)[5:-5, 5:-5]
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def _local_risk_map(source, result, output_width):
    """Check the actual candidate, after the preventive source mask was applied."""
    height = len(result)
    grid_h = max(1, math.ceil(height / 64))
    grid_w = 2 * grid_h
    boundaries = np.linspace(0, height, grid_h + 1).astype(int)
    risk = np.zeros((grid_h, grid_w), dtype=np.float32)
    totals = {key: 0 for key in ("missing_edge_blocks", "new_edge_blocks", "texture_gain_blocks",
                                "texture_loss_blocks", "new_flat_texture_blocks", "highlight_change_blocks")}
    for row, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        base = _resample_rows(source, output_width, start - 1, end + 1)
        candidate = _resample_rows(result, output_width, start - 1, end + 1)
        source_high, source_edge = _features(_luma(base))
        result_high, result_edge = _features(_luma(candidate))
        block_area = (end - start) * output_width / grid_w

        def mean(values):
            return cv2.resize(values.astype(np.float32), (grid_w, 1), interpolation=cv2.INTER_AREA)[0]

        se, re = source_edge >= 0.020, result_edge >= 0.020
        edge_count = mean(se) * block_area
        source_rms = np.sqrt(mean(source_high * source_high))
        result_rms = np.sqrt(mean(result_high * result_high))
        has_texture = source_rms > 0.0003
        flat = (source_edge < 0.005) & (np.abs(source_high) < 0.001)
        highlight_changed = (np.max(base[1:-1], axis=2) >= 0.96) & (np.max(np.abs(base[1:-1] - candidate[1:-1]), axis=2) > 1e-6)
        flags = {
            "missing_edge_blocks": mean(se & ~re) * block_area >= np.maximum(2, edge_count * 0.02),
            "new_edge_blocks": mean(~se & re) * block_area >= np.maximum(2, edge_count * 0.02),
            "texture_gain_blocks": has_texture & (result_rms > source_rms * 1.015 + 2e-6),
            "texture_loss_blocks": has_texture & (result_rms < source_rms * 0.97 - 2e-6),
            "new_flat_texture_blocks": mean(flat & (np.abs(result_high) > 0.003)) * block_area >= 1,
            "highlight_change_blocks": mean(highlight_changed) * block_area >= 0.5,
        }
        for name, flagged in flags.items():
            totals[name] += int(flagged.sum())
            risk[row] = np.maximum(risk[row], flagged.astype(np.float32))
    totals.update({"grid_dimensions": [grid_w, grid_h], "flagged_blocks": int(np.count_nonzero(risk)),
                   "total_blocks": int(risk.size), "scope": "实际候选相对同尺寸源图的分块风险检测，不等于物体语义判定。"})
    return risk, totals


def _restore_local(source, result, mask, raw_risk, output_width):
    """Restore risky blocks completely, with a smooth surrounding transition."""
    height = len(result)
    grid_h, grid_w = raw_risk.shape
    smooth_rgb = np.repeat(_smooth_protection(raw_risk)[..., None], 3, axis=2)
    x_bounds = np.linspace(0, output_width, grid_w + 1).astype(int)
    y_bounds = np.linspace(0, height, grid_h + 1).astype(int)
    x_index = np.searchsorted(x_bounds, np.arange(output_width), side="right") - 1
    alpha_sum = 0.0
    affected = 0
    for start in range(0, height, STRIP_ROWS):
        end = min(start + STRIP_ROWS, height)
        y_index = np.searchsorted(y_bounds, np.arange(start, end), side="right") - 1
        core = raw_risk[y_index[:, None], x_index[None, :]]
        smooth = _resample_rows(smooth_rgb, output_width, start, end)[..., 0]
        alpha = np.maximum(core, smooth)
        base = _resample_rows(source, output_width, start, end)
        result[start:end] += (base - result[start:end]) * alpha[..., None]
        # Exact assignment at the risk core avoids a floating-point residual.
        exact = core == 1
        result[start:end][exact] = base[exact]
        mask[start:end] = 1.0 - (1.0 - mask[start:end]) * (1.0 - alpha)
        alpha_sum += float(np.sum(alpha, dtype=np.float64))
        affected += int(np.count_nonzero(alpha > 0.001))
    return {"mean_blend": alpha_sum / (height * output_width), "affected_pixels": affected,
            "method": "风险块源像素完整回退，邻域平滑过渡；合成后重新审查，仍异常则整次明暗回退。"}


def _lookup(lut, values):
    position = np.clip(values, 0, 1) * (len(lut) - 1)
    index = np.minimum(position.astype(np.int32), len(lut) - 2)
    fraction = position - index
    return (lut[index] + fraction * (lut[index + 1] - lut[index])).astype(np.float32)


def _audit(source, result, output_width):
    counts = {"source_edge_pixels": 0, "result_edge_pixels": 0, "missing_edge_pixels": 0,
              "new_edge_pixels": 0, "source_flat_pixels": 0, "new_flat_texture_pixels": 0}
    source_energy = result_energy = difference = 0.0
    max_difference = highlight_difference = 0.0
    for start in range(0, len(result), STRIP_ROWS):
        end = min(start + STRIP_ROWS, len(result))
        base = _resample_rows(source, output_width, start - 1, end + 1)
        candidate = _resample_rows(result, output_width, start - 1, end + 1)
        source_high, source_edge = _features(_luma(base))
        result_high, result_edge = _features(_luma(candidate))
        se, re = source_edge >= 0.020, result_edge >= 0.020
        flat = (source_edge < 0.005) & (np.abs(source_high) < 0.001)
        counts["source_edge_pixels"] += int(se.sum())
        counts["result_edge_pixels"] += int(re.sum())
        counts["missing_edge_pixels"] += int((se & ~re).sum())
        counts["new_edge_pixels"] += int((~se & re).sum())
        counts["source_flat_pixels"] += int(flat.sum())
        counts["new_flat_texture_pixels"] += int((flat & (np.abs(result_high) > 0.003)).sum())
        source_energy += float(np.sum(source_high.astype(np.float64) ** 2))
        result_energy += float(np.sum(result_high.astype(np.float64) ** 2))
        error = np.abs(base[1:-1] - candidate[1:-1])
        difference += float(np.sum(error, dtype=np.float64))
        max_difference = max(max_difference, float(np.max(error)))
        highlights = np.max(base[1:-1], axis=2) >= 0.96
        if np.any(highlights):
            highlight_difference = max(highlight_difference, float(np.max(error[highlights])))
    pixels = result.shape[0] * result.shape[1]
    edge_total = max(counts["source_edge_pixels"], 1)
    counts.update({"missing_edge_fraction": counts["missing_edge_pixels"] / edge_total,
                   "new_edge_fraction": counts["new_edge_pixels"] / edge_total,
                   "high_frequency_rms_ratio": math.sqrt(result_energy / source_energy) if source_energy > 1e-10 else None,
                   "mean_rgb_change": difference / (pixels * 3), "maximum_rgb_change": max_difference,
                   "highlight_maximum_rgb_change": highlight_difference,
                   "scope": "相对同尺寸原图重采样基准；边缘阈值和高频能量仅为风险筛查，不是几何或材质语义验收。"})
    return counts


def _risk_reasons(metrics):
    reasons = []
    if metrics["missing_edge_pixels"] > max(64, metrics["source_edge_pixels"] * 0.02):
        reasons.append("missing_edges")
    if metrics["new_edge_pixels"] > max(64, metrics["source_edge_pixels"] * 0.02):
        reasons.append("new_edges")
    ratio = metrics["high_frequency_rms_ratio"]
    if ratio is not None and (ratio < 0.97 or ratio > 1.015):
        reasons.append("texture_energy_change")
    if metrics["new_flat_texture_pixels"] > max(32, metrics["source_flat_pixels"] * 0.0001):
        reasons.append("new_texture_on_flat_source")
    if metrics["highlight_maximum_rgb_change"] > 1e-6:
        reasons.append("highlight_change")
    return reasons


def process_panorama(source_rgb, guide_rgb, *, output_width, strength=0.35, analysis_width=1024):
    """Return (original-derived RGB, source-fallback mask, JSON-safe QA report).

    mask=1 means retain original tone, mask=0 permits the bounded LUT. The mask
    does not mean that geometry was repaired. No guide pixels enter the output.
    A failed final post-mask risk check discards the complete tone adjustment.
    """
    _validate_image(source_rgb, "原图")
    _validate_image(guide_rgb, "AI 明暗参考")
    output_width = _validate_width(output_width)
    analysis_width = _validate_width(analysis_width)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("明暗参考强度必须在 0–1。")
    source_small = _analysis_image(source_rgb, analysis_width)
    guide_small = _analysis_image(guide_rgb, analysis_width)
    lut, tone_report = build_tone_lut(source_small, guide_small, strength)
    work_source = source_rgb
    if output_width < source_rgb.shape[1]:
        work_source = resize_equirectangular(source_rgb, output_width)
    output_height = output_width // 2
    result = np.empty((output_height, output_width, 3), dtype=np.float32)
    mask = np.empty((output_height, output_width), dtype=np.float32)
    preventive_sum = 0.0
    for start in range(0, output_height, STRIP_ROWS):
        end = min(start + STRIP_ROWS, output_height)
        base = _resample_rows(work_source, output_width, start, end)
        if tone_report["identity"]:
            result[start:end] = base
            mask[start:end] = 1.0
            continue
        peak = np.max(base, axis=2)
        # Protect resolved line pixels narrowly; fine texture alone is not a
        # reason to suppress the room's complete tone adjustment.
        edge_protection = _edge_protection_rows(work_source, output_width, start, end)
        alpha = (1.0 - edge_protection) * (1.0 - _smoothstep(peak, 0.82, 0.96))
        luminance = _luma(base)
        mapped = _lookup(lut, luminance)
        ratio = np.divide(mapped, luminance, out=np.ones_like(mapped), where=luminance > 1e-7)
        gain = 1.0 + alpha * (ratio - 1.0)
        # Scaling all channels together preserves RGB ratios; never clip channels.
        gain = np.minimum(gain, np.divide(1.0, peak, out=np.ones_like(peak), where=peak > 0))
        result[start:end] = base * gain[..., None]
        mask[start:end] = 1.0 - alpha
        preventive_sum += float(np.sum(1.0 - alpha, dtype=np.float64))
    if tone_report["identity"]:
        local_report = {"checked": False, "applied": False, "reason": "identity_tone", "flagged_blocks": 0}
        before_local = None
    else:
        raw_risk, local_report = _local_risk_map(work_source, result, output_width)
        local_report.update({"checked": True, "applied": bool(local_report["flagged_blocks"])})
        before_local = None
        if local_report["applied"]:
            before_local = _audit(work_source, result, output_width)
            local_report.update(_restore_local(work_source, result, mask, raw_risk, output_width))
    metrics = _audit(work_source, result, output_width)
    reasons = _risk_reasons(metrics)
    before_fallback = None
    if reasons:
        before_fallback = metrics
        for start in range(0, output_height, STRIP_ROWS):
            end = min(start + STRIP_ROWS, output_height)
            result[start:end] = _resample_rows(work_source, output_width, start, end)
        mask.fill(1.0)
        metrics = _audit(work_source, result, output_width)
    guide_at_source_analysis = resize_equirectangular(guide_small, source_small.shape[1])
    guide_metrics = _audit(source_small, guide_at_source_analysis, source_small.shape[1])
    report = {"algorithm": "source_geometry_histogram_tone_v4", "source_dimensions": list(source_rgb.shape[1::-1]),
              "guide_dimensions": list(guide_rgb.shape[1::-1]), "output_dimensions": [output_width, output_height],
              "guide_role": "仅亮度分位统计；不使用 AI 空间坐标、边缘、纹理或 RGB 像素。",
              "resampling": "同尺寸原样；缩小整图面积重采样；放大周期经度双线性、跨极点半圈经度映射。",
              "tone": tone_report, "fallback_mask_mean": float(np.mean(mask, dtype=np.float64)),
              "source_preventive_protection": {"active": not tone_report["identity"],
                                               "mean_mask": preventive_sum / (output_height * output_width),
                                               "scope": "源高光和明确细线周围窄边带的预防保护；不会因任意细粒纹理保护整片房间。实际纹理风险另行检测。"},
              "detected_local_fallback": local_report,
              "candidate_metrics_before_local_fallback": before_local,
              "global_tone_fallback": bool(reasons), "global_fallback_reasons": reasons,
              "candidate_metrics_before_global_fallback": before_fallback, "final_metrics": metrics,
              "guide_diagnostics_only": guide_metrics,
              "limitations": ["保留原图已有的污渍、颗粒、缺陷和水印，不会凭空恢复未记录的细节。",
                              "本方法保留源几何，仅提供受限明暗参考，可能比重新生成的摄影感弱。",
                              "检查通过不等于所有全景视觉验收通过；高频和边缘统计不理解物体语义。"]}
    return result, mask, report
