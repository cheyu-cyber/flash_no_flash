"""Translation alignment between two RGB images via template matching.

Adapted from FlowECT/interpolate_shift.py. Used to register flash / no-flash
real-photo pairs that are slightly mis-aligned because the camera moved
between shots — synthetic pairs are pixel-perfect and don't need this.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """RGB float [0,1] or grayscale -> grayscale uint8."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255.0).astype(np.uint8)
    return gray


def _auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """ Canny with thresholds derived from the per-image median. """
    med = float(np.median(img))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img, lo, hi)


def estimate_shift(
    ref: np.ndarray,
    target: np.ndarray,
    template_size: int = 512,
    use_edges: bool = True,
) -> tuple[float, float, float]:
    """Return (dx, dy, score) such that ``ref`` shifted by (dx, dy) ≈ ``target``.

    Crops a central template from ``ref`` and locates it inside ``target`` via
    normalised cross-correlation. ``use_edges=True`` runs Canny first, which
    is insensitive to absolute intensity and recommended for flash / no-flash
    pairs (their brightness and colour differ a lot).
    """
    A = _to_gray_uint8(ref)
    B = _to_gray_uint8(target)

    H, W = A.shape
    th = min(template_size, H - 2)
    tw = min(template_size, W - 2)
    y0 = (H - th) // 2
    x0 = (W - tw) // 2

    if use_edges:
        A = _auto_canny(A)
        B = _auto_canny(B)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    template = A[y0:y0 + th, x0:x0 + tw]
    res = cv2.matchTemplate(B, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(res)
    return float(px - x0), float(py - y0), float(score)


def align_pair_and_crop(
    flash: np.ndarray,
    noflash: np.ndarray,
    template_size: int = 512,
    use_edges: bool = True,
    max_shift_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[float, float, float, bool]]:
    """Translate ``noflash`` to align with ``flash`` and crop both to the mutually-valid region.

    Steps:

    1. Estimate (dx, dy) such that ``flash`` shifted by (+dx, +dy) ≈ ``noflash``.
    2. Warp ``noflash`` by (-dx, -dy) — same scene at flash's coordinates,
       but with a (|dx|, |dy|) border of synthetic pixels on the side the
       warp brought in from outside the source.
    3. Crop both images to the rectangle that excludes that border. A
       1-pixel shift therefore strips a 1-pixel border from both images so
       the pair stays pixel-aligned and the same shape.

    ``max_shift_frac`` rejects detections where ``|dx|/W`` or ``|dy|/H``
    exceeds the threshold. Edge-Canny matching on flash/no-flash pairs
    sometimes locks onto a uniform region and reports a huge confident
    shift; without this guard a single bad match would crop most of the
    image away. Rejected pairs are returned uncropped with dx=dy=0.

    Returns ``(flash_crop, noflash_crop, crop_box, (dx, dy, score, applied))``
    where ``crop_box = (top, bottom, left, right)`` so callers can apply the
    same crop to a third image (e.g. a Petschnigg reference at the same
    native resolution). ``applied`` is False when the safety check vetoed
    the warp.
    """
    dx, dy, score = estimate_shift(flash, noflash, template_size, use_edges)
    H, W = flash.shape[:2]

    if abs(dx) / W > max_shift_frac or abs(dy) / H > max_shift_frac:
        return flash, noflash, (0, H, 0, W), (dx, dy, score, False)

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(noflash, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # warpAffine samples output(u, v) from input(u + dx, v + dy):
    #   dx > 0 → right `dx` cols are out-of-source (filled), so crop them.
    #   dx < 0 → left `|dx|` cols are out-of-source.
    # Same for dy with top/bottom rows.
    dxi, dyi = int(round(dx)), int(round(dy))
    top = max(0, -dyi)
    bottom = H - max(0, dyi)
    left = max(0, -dxi)
    right = W - max(0, dxi)
    crop_box = (top, bottom, left, right)

    flash_crop = flash[top:bottom, left:right]
    noflash_crop = aligned[top:bottom, left:right]
    return flash_crop, noflash_crop, crop_box, (dx, dy, score, True)
