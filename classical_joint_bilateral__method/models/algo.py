"""Core filters from Petschnigg et al. (2004) for flash / no-flash denoising.

This file implements the two filters used in the denoising section:

1. Bilateral filter, corresponding to Eqs. (2) and (3).
2. Joint bilateral filter, corresponding to Eq. (4).

The implementation is intentionally direct and readable, so it mirrors the paper's
notation rather than being the fastest possible version.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

Array = np.ndarray



def _ensure_float_image(image: Array) -> Array:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    return image



def _to_channel_last(image: Array) -> tuple[Array, bool]:
    """Return HxWxC image and whether the input was grayscale."""
    image = _ensure_float_image(image)
    if image.ndim == 2:
        return image[..., None], True
    return image, False



def _from_channel_last(image: Array, squeeze_gray: bool) -> Array:
    if squeeze_gray:
        return image[..., 0]
    return image



def _gaussian(x: Array | float, sigma: float) -> Array | float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return np.exp(-(np.asarray(x) ** 2) / (2.0 * sigma * sigma))



def _default_radius(sigma_d: float) -> int:
    return max(1, int(np.ceil(3.0 * sigma_d)))



def _spatial_kernel(radius: int, sigma_d: float) -> Array:
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    dist2 = xs * xs + ys * ys
    kernel = np.exp(-dist2 / (2.0 * sigma_d * sigma_d))
    return np.asarray(kernel, dtype=np.float64)



def _reflect_pad(image: Array, radius: int) -> Array:
    if image.ndim == 2:
        return np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    return np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")



def _range_distance(center: Array, neighborhood: Array) -> Array:
    """Return scalar intensity or RGB Euclidean difference per neighbor."""
    diff = neighborhood - center
    if diff.ndim == 2:
        return np.abs(diff)
    return np.linalg.norm(diff, axis=-1)



def bilateral_filter(
    image: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
) -> Array:
    """Apply the paper's bilateral filter to an image.

    Equation mapping
    ----------------
    Implements Eq. (2):
        A_base(p) = (1 / k(p)) * sum_{p' in Omega} g_d(p' - p) g_r(A_{p'} - A_p) A_{p'}

    and Eq. (3):
        k(p) = sum_{p' in Omega} g_d(p' - p) g_r(A_{p'} - A_p)

    Parameters
    ----------
    image:
        Ambient image A in the paper. Can be HxW or HxWxC.
    sigma_d:
        Spatial standard deviation for g_d.
    sigma_r:
        Range / intensity standard deviation for g_r.
    radius:
        Local window radius. Defaults to ceil(3*sigma_d).
    """
    image_cl, squeeze_gray = _to_channel_last(image)
    h, w, c = image_cl.shape
    radius = _default_radius(sigma_d) if radius is None else int(radius)

    spatial = _spatial_kernel(radius, sigma_d)
    padded = _reflect_pad(image_cl, radius)
    out = np.zeros_like(image_cl)

    for y in range(h):
        for x in range(w):
            center = padded[y + radius, x + radius]
            patch = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]

            dist_range = _range_distance(center, patch)
            range_w = np.asarray(_gaussian(dist_range, sigma_r), dtype=np.float64)
            weights = spatial * range_w
            norm = np.sum(weights)
            if norm <= 1e-12:
                out[y, x] = center
                continue

            weighted_patch = patch * weights[..., None]
            out[y, x] = np.sum(weighted_patch, axis=(0, 1)) / norm

    return _from_channel_last(out, squeeze_gray)



def joint_bilateral_filter(
    ambient: Array,
    flash: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
) -> Array:
    """Apply the paper's joint bilateral filter.

    Equation mapping
    ----------------
    Implements Eq. (4):
        A_NR(p) = (1 / k(p)) * sum_{p' in Omega} g_d(p' - p) g_r(F_{p'} - F_p) A_{p'}

    The ambient image A is averaged, but the range weight g_r is computed from
    the flash image F, exactly as described in the paper.
    """
    ambient_cl, squeeze_gray = _to_channel_last(ambient)
    flash_cl, _ = _to_channel_last(flash)

    if ambient_cl.shape != flash_cl.shape:
        raise ValueError("ambient and flash must have the same shape")

    h, w, _ = ambient_cl.shape
    radius = _default_radius(sigma_d) if radius is None else int(radius)

    spatial = _spatial_kernel(radius, sigma_d)
    ambient_pad = _reflect_pad(ambient_cl, radius)
    flash_pad = _reflect_pad(flash_cl, radius)
    out = np.zeros_like(ambient_cl)

    for y in range(h):
        for x in range(w):
            flash_center = flash_pad[y + radius, x + radius]
            flash_patch = flash_pad[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            ambient_patch = ambient_pad[y : y + 2 * radius + 1, x : x + 2 * radius + 1]

            dist_range = _range_distance(flash_center, flash_patch)
            range_w = np.asarray(_gaussian(dist_range, sigma_r), dtype=np.float64)
            weights = spatial * range_w
            norm = np.sum(weights)
            if norm <= 1e-12:
                out[y, x] = ambient_pad[y + radius, x + radius]
                continue

            weighted_patch = ambient_patch * weights[..., None]
            out[y, x] = np.sum(weighted_patch, axis=(0, 1)) / norm

    return _from_channel_last(out, squeeze_gray)


__all__ = [
    "bilateral_filter",
    "joint_bilateral_filter",
]
