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
    h, w, c = image.shape
    radius = _default_radius(sigma_d) if radius is None else int(radius)
    # g_d(p' - p), a precomputed 2D Gaussian kernel based on σ_d
    spatial = _spatial_kernel(radius, sigma_d) 
    # # A_{p'} for p' in Omega
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            # A_p, the center pixel value at (y, x)
            center = padded[y + radius, x + radius] 
            # A_{p'}, the pixel values in the neighborhood around (y, x)
            patch = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1] 
            # A_{p'} - A_p
            diff = patch - center 
            dist_range = np.linalg.norm(diff, axis=-1)
            # g_r(A_{p'} - A_p), the range weights based on intensity differences
            range_w = np.asarray(_gaussian(dist_range, sigma_r), dtype=np.float64) 
            # g_d(p' - p) * g_r(A_{p'} - A_p), the combined spatial and range weights
            weights = spatial * range_w
            # k(p)
            norm = np.sum(weights) 
            if norm <= 1e-12:
                out[y, x] = center
                continue
            # g_d(p' - p) * g_r(A_{p'} - A_p) * A_{p'}, the weighted pixel values in the neighborhood
            weighted_patch = patch * weights[..., None] 
            # (1 / k(p)) * sum_{p' in Omega} g_d(p' - p) g_r(A_{p'} - A_p) A_{p'}
            out[y, x] = np.sum(weighted_patch, axis=(0, 1)) / norm 
    return out

def bilateral_filter_luminance(
    image: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
) -> Array:
    """Apply the paper's bilateral filter to the luminance channel only.

    Works for any channel-last 3-channel image whose **first channel** is
    luminance — i.e. YCbCr (Y), Lab (L), or YUV (Y). The two chroma channels
    are passed through unchanged. Caller is responsible for the RGB <-> target
    conversion.
    """
    y_filtered = bilateral_filter(image[..., :1], sigma_d, sigma_r, radius)
    out = image.copy()
    out[..., :1] = y_filtered
    return out

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
    

    if ambient.shape != flash.shape:
        raise ValueError("ambient and flash must have the same shape")

    h, w, _ = ambient.shape
    radius = _default_radius(sigma_d) if radius is None else int(radius)
    # g_d(p' - p), a precomputed 2D Gaussian kernel based on σ_d
    spatial = _spatial_kernel(radius, sigma_d)
    # A_{p'} for p' in Omega and F_{p'} for p' in Omega
    ambient_pad = np.pad(ambient, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    flash_pad = np.pad(flash, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(ambient)

    for y in range(h):
        for x in range(w):
            # F_p, the center pixel value in the flash image at (y, x)
            flash_center = flash_pad[y + radius, x + radius]
            # F_{p'}, the pixel values in the neighborhood around (y, x)
            flash_patch = flash_pad[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            # A_{p'}, the pixel values in the neighborhood around (y, x)
            ambient_patch = ambient_pad[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            # F_{p'} - F_p
            diff = flash_patch - flash_center
            dist_range = np.linalg.norm(diff, axis=-1)
            # g_r(F_{p'} - F_p), the range weights based on intensity differences
            range_w = np.asarray(_gaussian(dist_range, sigma_r), dtype=np.float64)
            # g_d(p' - p) * g_r(F_{p'} - F_p)
            weights = spatial * range_w
            # k(p)
            norm = np.sum(weights)
            if norm <= 1e-12:
                out[y, x] = ambient_pad[y + radius, x + radius]
                continue
            # g_d(p' - p) * g_r(F_{p'} - F_p) * A_{p'}, the weighted pixel values in the neighborhood
            weighted_patch = ambient_patch * weights[..., None]
            # (1 / k(p)) * sum_{p' in Omega} g_d(p' - p) g_r(F_{p'} - F_p) A_{p'}
            out[y, x] = np.sum(weighted_patch, axis=(0, 1)) / norm

    return out

def joint_bilateral_filter_luminance(
    ambient: Array,
    flash: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
) -> Array:
    """Apply the paper's joint bilateral filter to the luminance channel only.

    Works for any channel-last 3-channel image whose first channel is
    luminance (YCbCr, Lab, YUV). Chroma channels of `ambient` pass through.
    """
    y_filtered = joint_bilateral_filter(
        ambient[..., :1], flash[..., :1], sigma_d, sigma_r, radius
    )
    out = ambient.copy()
    out[..., :1] = y_filtered
    return out


__all__ = [
    "bilateral_filter",
    "bilateral_filter_luminance",
    "joint_bilateral_filter",
    "joint_bilateral_filter_luminance",
]
