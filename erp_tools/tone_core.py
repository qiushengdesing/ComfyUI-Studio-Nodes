import numpy as np
import cv2
from scipy.fft import dct,idct

def harmonic_field(boundary, L):
    h = boundary.shape[0]
    a = np.arccosh(2 - np.cos(np.pi * np.arange(h, dtype=np.float64) / h))
    x = np.arange(L + 1, dtype=np.float64)
    ratio = np.empty((h, L + 1), np.float64)
    ratio[0] = 1 - x / L
    aa = a[1:, None]
    ratio[1:] = np.exp(-aa * x) * -np.expm1(-2 * aa * (L - x)) / -np.expm1(-2 * aa * L)
    return idct(dct(boundary, type=2, norm='ortho', axis=0)[:, None, :] * ratio[:, :, None], type=2, norm='ortho', axis=0)

def apply_field(f, field):
    L = field.shape[1] - 1
    out = f.copy()
    out[:, :L + 1] -= field
    out[:, -L - 1:] += field[:, ::-1]
    return out

def dual_harmonic(im, wide_fraction=0.09, narrow_width=6):
    h, w = im.shape[:2]
    L = round(w * wide_fraction)
    near = max(2, round(w * 6 / 4500))
    far = max(near + 4, round(w * 16 / 4500))
    f = im.astype(np.float64)
    broad_jump = np.median(f[:, near:far], axis=1) - np.median(f[:, -far:-near], axis=1)
    broad_jump = cv2.GaussianBlur(broad_jump[:, None, :], (1, 0), 0, sigmaY=max(0.7, h * 1.5 / 2250), borderType=cv2.BORDER_REPLICATE)[:, 0, :]
    broad = harmonic_field(broad_jump / 2, L)
    wide = apply_field(f, broad)
    residual = wide[:, 0] - wide[:, -1]
    narrow = harmonic_field(residual / 2, narrow_width)
    corrected = apply_field(wide, narrow)
    out = np.rint(np.clip(corrected, 0, 255)).astype(np.uint8)
    endpoint = np.rint(np.clip((wide[:, 0] + wide[:, -1]) / 2, 0, 255)).astype(np.uint8)
    out[:, 0] = endpoint
    out[:, -1] = endpoint
    return (out, {'wide_band_each_side': L, 'robust_sample_range': [near, far], 'narrow_band_each_side': narrow_width, 'broad_jump_max_abs': float(np.abs(broad_jump).max()), 'residual_jump_max_abs': float(np.abs(residual).max()), 'endpoint_rgb_mae': float(np.abs(out[:, 0].astype(np.int16) - out[:, -1].astype(np.int16)).mean()), 'outside_wide_band_max_difference': int(np.abs(out[:, L:-L].astype(np.int16) - im[:, L:-L].astype(np.int16)).max()), 'clipped_channel_fraction': float(np.mean((corrected < 0) | (corrected > 255))), 'vertical_boundary': 'DCT-II Neumann; no top-bottom wrap'})