import numpy as np
import cv2
import torch
import time

def load_lama_cpu(model_path):
    with open(model_path, 'rb') as handle:
        return torch.jit.load(handle, map_location='cpu').eval()

def finish_panorama_lama_cpu(image, model_path=None, mask_width=8, context=768, overlap=128, model=None, progress=None):
    """Finish one uint8 HWC panorama with overlapping CPU LaMa tiles.

    mask_width is the TOTAL native-resolution strip width across the seam,
    shared equally by the image's left and right edges. No global resizing is
    performed. Returns the image and factual preservation/runtime metrics.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must be a uint8 RGB array')
    height, width = image.shape[:2]
    if mask_width <= 0 or mask_width % 2 or mask_width >= width:
        raise ValueError('mask_width must be positive, even, and below image width')
    if context % 8 or context > width or context <= mask_width or (not 0 < overlap < context):
        raise ValueError('context must be divisible by 8 and fit width; overlap must fit context')
    started = time.perf_counter()
    if model is None:
        if model_path is None:
            raise ValueError('model_path or an already loaded CPU model is required')
        model = load_lama_cpu(model_path)
    model = model.cpu().eval()
    middle = width // 2
    rolled = np.roll(image, middle, axis=1)
    strip_start, strip_end = (middle - mask_width // 2, middle + mask_width // 2)
    crop_start = middle - context // 2
    accum = np.zeros((height, mask_width, 3), np.float32)
    weight_sum = np.zeros((height, 1, 1), np.float32)
    starts = list(range(0, max(1, height - context + 1), context - overlap))
    final_start = max(0, height - context)
    if starts[-1] != final_start:
        starts.append(final_start)
    mask = torch.zeros((1, 1, context, context), dtype=torch.float32)
    mask[:, :, :, context // 2 - mask_width // 2:context // 2 + mask_width // 2] = 1
    for number, y0 in enumerate(starts):
        y1 = min(height, y0 + context)
        valid = y1 - y0
        patch = rolled[y0:y1, crop_start:crop_start + context].copy()
        if valid < context:
            patch = cv2.copyMakeBorder(patch, 0, context - valid, 0, 0, cv2.BORDER_REFLECT_101)
        tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0)
        with torch.no_grad():
            generated = model(tensor, mask)[0].permute(1, 2, 0).cpu().numpy()
        strip = generated[:valid, context // 2 - mask_width // 2:context // 2 + mask_width // 2]
        ramp = np.ones(valid, np.float32)
        if number > 0:
            n = min(valid, starts[number - 1] + context - y0)
            ramp[:n] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(n, dtype=np.float32) + 0.5) / n)
        if number + 1 < len(starts):
            n = min(valid, y1 - starts[number + 1])
            ramp[-n:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n, dtype=np.float32) + 0.5) / n)
        accum[y0:y1] += strip * ramp[:, None, None]
        weight_sum[y0:y1] += ramp[:, None, None]
        if progress:
            progress({'mask_width': mask_width, 'tile': number + 1, 'tile_count': len(starts), 'y0': y0, 'y1': y1})
    if not np.all(weight_sum > 0):
        raise RuntimeError('tile blending left an uncovered row')
    result = rolled.copy()
    result[:, strip_start:strip_end] = np.rint(np.clip(accum / weight_sum, 0, 1) * 255).astype(np.uint8)
    restored = np.roll(result, -middle, axis=1)
    delta = np.abs(result.astype(np.int16) - rolled.astype(np.int16))
    horizontal = np.abs(np.diff(result.astype(np.float32), axis=1)).mean(axis=(0, 2))
    metrics = {'device': 'cpu', 'native_size': [width, height], 'mask_width_total': mask_width, 'context': context, 'tile_starts': starts, 'minimum_blending_weight': float(weight_sum.min()), 'runtime_seconds': time.perf_counter() - started, 'outside_mask_max_difference': int(max(delta[:, :strip_start].max(), delta[:, strip_end:].max())), 'changed_pixel_fraction': float(np.any(delta != 0, axis=2).mean()), 'seam_adjacent_rgb_mae_255': float(horizontal[middle - 1]), 'left_mask_boundary_rgb_mae_255': float(horizontal[strip_start - 1]), 'right_mask_boundary_rgb_mae_255': float(horizontal[strip_end - 1]), 'mask_mean_abs_change_255': float(delta[:, strip_start:strip_end].mean()), 'limitation': 'Narrow content completion; cannot fix geometry outside the mask.'}
    return (restored, metrics)