"""Denoising pipeline structure that follows Figure 3 from the paper.

This file mirrors the denoising-side overview in Figure 3 of:
Petschnigg et al., "Digital Photography with Flash and No-Flash Image Pairs".

The flow is:
    A (ambient / no-flash) -> bilateral filter -> A_base
    A + F (flash) -> joint bilateral filter -> A_NR
    linearized A and F -> shadow/specularity detection -> M
    final merge -> A_NR_prime = (1 - M) * A_NR + M * A_base

The implementation below is lightweight and intended as a clear structural match
for the paper, not as a production-optimized image processing library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .algo import bilateral_filter, joint_bilateral_filter
except ImportError:
    from models.algo import bilateral_filter, joint_bilateral_filter

Array = np.ndarray


@dataclass
class DenoisingResult:
    ambient_base: Array
    ambient_joint: Array
    shadow_mask: Array
    specularity_mask: Array
    artifact_mask: Array
    denoised: Array



def _ensure_float_image(image: Array) -> Array:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    return image



def _to_channel_last(image: Array) -> tuple[Array, bool]:
    image = _ensure_float_image(image)
    if image.ndim == 2:
        return image[..., None], True
    return image, False



def _from_channel_last(image: Array, squeeze_gray: bool) -> Array:
    if squeeze_gray:
        return image[..., 0]
    return image



def _rgb_luminance(image: Array) -> Array:
    image_cl, squeeze_gray = _to_channel_last(image)
    if squeeze_gray:
        return image_cl[..., 0]
    if image_cl.shape[-1] == 1:
        return image_cl[..., 0]
    r = image_cl[..., 0]
    g = image_cl[..., 1]
    b = image_cl[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b



def _max_filter(mask: Array, radius: int) -> Array:
    if radius <= 0:
        return mask.copy()
    mask = np.asarray(mask, dtype=np.float64)
    h, w = mask.shape
    padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="edge")
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            patch = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            out[y, x] = np.max(patch)
    return out



def _min_filter(mask: Array, radius: int) -> Array:
    if radius <= 0:
        return mask.copy()
    mask = np.asarray(mask, dtype=np.float64)
    h, w = mask.shape
    padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="edge")
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            patch = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]
            out[y, x] = np.min(patch)
    return out



def _binary_open_close(mask: Array, radius: int = 1) -> Array:
    mask = (mask > 0.5).astype(np.float64)
    eroded = _min_filter(mask, radius)
    opened = _max_filter(eroded, radius)
    dilated = _max_filter(opened, radius)
    closed = _min_filter(dilated, radius)
    return (closed > 0.5).astype(np.float64)



def _feather_mask(mask: Array, radius: int = 2) -> Array:
    mask = np.asarray(mask, dtype=np.float64)
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0)
    size = 2 * radius + 1
    padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="edge")
    out = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            patch = padded[y : y + size, x : x + size]
            out[y, x] = np.mean(patch)
    return np.clip(out, 0.0, 1.0)



def linearize_ambient_to_flash_space(
    ambient_linear_prime: Array,
    iso_ambient: float,
    t_ambient: float,
    iso_flash: float,
    t_flash: float,
) -> Array:
    """Exposure-normalize A' into the same linear space as F.

    Equation mapping
    ----------------
    Implements Eq. (1):
        A_lin = A'_lin * (ISO_F * Delta t_F) / (ISO_A * Delta t_A)
    """
    ambient_linear_prime = _ensure_float_image(ambient_linear_prime)
    scale = (iso_flash * t_flash) / (iso_ambient * t_ambient)
    return ambient_linear_prime * scale



def detect_shadow_mask(
    ambient_linear: Array,
    flash_linear: Array,
    tau_shadow: float,
    morph_radius: int = 1,
    feather_radius: int = 2,
) -> Array:
    """Detect flash shadows using the paper's threshold rule.

    Equation mapping
    ----------------
    Implements Eq. (8):
        M_shad = 1 when |F_lin - A_lin| <= tau_shad, else 0

    We apply the test to luminance, then do simple morphology and feathering to
    mimic the paper's cleanup description.
    """
    a_l = _rgb_luminance(ambient_linear)
    f_l = _rgb_luminance(flash_linear)
    shadow = (np.abs(f_l - a_l) <= tau_shadow).astype(np.float64)
    shadow = _binary_open_close(shadow, radius=morph_radius)
    shadow = _max_filter(shadow, morph_radius)  # conservative dilation
    shadow = _feather_mask(shadow, feather_radius)
    return shadow



def detect_specularity_mask(
    flash_linear: Array,
    saturation_threshold: float = 0.95,
    morph_radius: int = 1,
    feather_radius: int = 2,
) -> Array:
    """Detect flash specularities with the paper's heuristic.

    Formula / heuristic mapping
    ---------------------------
    Section 4.3 states that specular regions are detected where flash luminance is
    greater than 95% of the sensor range. This is not given an equation number, but
    it is one of the key rules in the denoising overview.
    """
    flash_l = _rgb_luminance(flash_linear)
    spec = (flash_l >= saturation_threshold).astype(np.float64)
    spec = _binary_open_close(spec, radius=morph_radius)
    spec = _max_filter(spec, morph_radius)
    spec = _feather_mask(spec, feather_radius)
    return spec



def detect_flash_artifact_mask(
    ambient_linear: Array,
    flash_linear: Array,
    tau_shadow: float,
    saturation_threshold: float = 0.95,
    morph_radius: int = 1,
    feather_radius: int = 2,
) -> tuple[Array, Array, Array]:
    """Return shadow mask, specularity mask, and their union M."""
    shadow = detect_shadow_mask(
        ambient_linear,
        flash_linear,
        tau_shadow=tau_shadow,
        morph_radius=morph_radius,
        feather_radius=feather_radius,
    )
    spec = detect_specularity_mask(
        flash_linear,
        saturation_threshold=saturation_threshold,
        morph_radius=morph_radius,
        feather_radius=feather_radius,
    )
    merged = np.clip(np.maximum(shadow, spec), 0.0, 1.0)
    merged = _feather_mask(merged, feather_radius)
    return shadow, spec, merged



def merge_denoised_with_mask(
    ambient_joint: Array,
    ambient_base: Array,
    mask: Array,
) -> Array:
    """Final denoising merge from the paper.

    Equation mapping
    ----------------
    Implements Eq. (5):
        A'_NR = (1 - M) A_NR + M A_base
    """
    ambient_joint = _ensure_float_image(ambient_joint)
    ambient_base = _ensure_float_image(ambient_base)
    if ambient_joint.shape != ambient_base.shape:
        raise ValueError("ambient_joint and ambient_base must have the same shape")

    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 2 and ambient_joint.ndim == 3:
        mask = mask[..., None]
    if mask.shape[:2] != ambient_joint.shape[:2]:
        raise ValueError("mask spatial dimensions must match image")

    return (1.0 - mask) * ambient_joint + mask * ambient_base



def denoise_pipeline(
    ambient: Array,
    flash: Array,
    sigma_d: float = 3.0,
    sigma_r_bilateral: float = 0.08,
    sigma_r_joint: float = 0.001,
    ambient_linear: Optional[Array] = None,
    flash_linear: Optional[Array] = None,
    tau_shadow: float = 0.02,
    saturation_threshold: float = 0.95,
    radius: Optional[int] = None,
    morph_radius: int = 1,
    feather_radius: int = 2,
) -> DenoisingResult:
    """Run the denoising-side Figure 3 pipeline.

    Steps
    -----
    1. A_base  = bilateral_filter(A)
    2. A_NR    = joint_bilateral_filter(A, F)
    3. M       = shadow/specularity mask from linearized images
    4. output  = (1 - M) * A_NR + M * A_base

    If ambient_linear or flash_linear are not provided, the function falls back to a
    zero artifact mask and returns the raw joint-bilateral result as the final output.
    """
    ambient = _ensure_float_image(ambient)
    flash = _ensure_float_image(flash)
    if ambient.shape != flash.shape:
        raise ValueError("ambient and flash must have the same shape")

    ambient_base = bilateral_filter(
        ambient,
        sigma_d=sigma_d,
        sigma_r=sigma_r_bilateral,
        radius=radius,
    )
    ambient_joint = joint_bilateral_filter(
        ambient,
        flash,
        sigma_d=sigma_d,
        sigma_r=sigma_r_joint,
        radius=radius,
    )

    if ambient_linear is None or flash_linear is None:
        shadow_mask = np.zeros(ambient.shape[:2], dtype=np.float64)
        specularity_mask = np.zeros(ambient.shape[:2], dtype=np.float64)
        artifact_mask = np.zeros(ambient.shape[:2], dtype=np.float64)
        denoised = ambient_joint.copy()
    else:
        shadow_mask, specularity_mask, artifact_mask = detect_flash_artifact_mask(
            ambient_linear=ambient_linear,
            flash_linear=flash_linear,
            tau_shadow=tau_shadow,
            saturation_threshold=saturation_threshold,
            morph_radius=morph_radius,
            feather_radius=feather_radius,
        )
        denoised = merge_denoised_with_mask(ambient_joint, ambient_base, artifact_mask)

    return DenoisingResult(
        ambient_base=ambient_base,
        ambient_joint=ambient_joint,
        shadow_mask=shadow_mask,
        specularity_mask=specularity_mask,
        artifact_mask=artifact_mask,
        denoised=denoised,
    )


__all__ = [
    "DenoisingResult",
    "linearize_ambient_to_flash_space",
    "detect_shadow_mask",
    "detect_specularity_mask",
    "detect_flash_artifact_mask",
    "merge_denoised_with_mask",
    "denoise_pipeline",
]
