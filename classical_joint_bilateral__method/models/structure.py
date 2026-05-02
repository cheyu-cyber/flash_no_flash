from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
Array = np.ndarray
from models.algo import bilateral_filter, joint_bilateral_filter


@dataclass
class DenoisingResult:
    ambient_base: Array
    ambient_joint: Array
    shadow_mask: Array
    specularity_mask: Array
    artifact_mask: Array
    result: Array

#---RGB to luminance and mask cleanup helpers (Section 4.3 prose)-------
def _to_channel_last(image: Array) -> tuple[Array, bool]:
    if image.ndim == 2:
        return image[..., None], True
    return image, False

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

#-----Mask cleanup helpers (Section 4.3 prose)-------
def _clean_binary_mask(mask: Array, radius: int = 1) -> Array:
    """Open -> close -> conservative dilate, per Section 4.3 prose."""
    if radius <= 0:
        return np.clip(mask.astype(np.float64), 0.0, 1.0)
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)

    binary = cv2.erode(binary, kernel)    # open  ...
    binary = cv2.dilate(binary, kernel)   #   ... removes speckles
    binary = cv2.dilate(binary, kernel)   # close ...
    binary = cv2.erode(binary, kernel)    #   ... fills pinholes
    binary = cv2.dilate(binary, kernel)   # conservative coverage

    return binary.astype(np.float64)

def _feather_mask(mask: Array, radius: int = 2) -> Array:
    """Gaussian blur to feather the mask edges (Section 4.3 final-merge prose)."""
    if radius <= 0:
        return np.clip(mask, 0.0, 1.0)
    sigma = radius / 1.5
    blurred = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma)
    return np.clip(blurred, 0.0, 1.0)

def detect_shadow_mask(
    ambient_linear: Array,
    flash_linear: Array,
    tau_shadow: float,
) -> Array:
    """Detect flash shadows"""
    a_l = _rgb_luminance(ambient_linear)
    f_l = _rgb_luminance(flash_linear)
    return (np.abs(f_l - a_l) <= tau_shadow).astype(np.float64)

def detect_specularity_mask(
    flash_linear: Array,
    saturation_threshold: float = 0.95,
) -> Array:
    """
    Section 4.3: specular regions are detected where flash luminance is greater
    than 95% of the sensor range. No equation number is given in the paper.
    """
    flash_l = _rgb_luminance(flash_linear)
    return (flash_l >= saturation_threshold).astype(np.float64)

def detect_flash_artifact_mask(
    ambient_linear: Array,
    flash_linear: Array,
    tau_shadow: float,
    saturation_threshold: float = 0.95,
    morph_radius: int = 1,
    feather_radius: int = 2,
) -> tuple[Array, Array, Array]:
    """Return shadow mask, specularity mask, and their union M.

    Section 4.3 describes the cleanup applied here:

    - Per individual mask: morphological (erode then dilate) to remove
      speckles and fill holes, then a final dilate for a conservative estimate.
    - On the union: blur to feather the edges so that the final blend is seamless.

    Eq. (8) and the 95% heuristic produce the raw masks; everything else here
    comes from the prose in Section 4.3.
    """
    shadow_raw = detect_shadow_mask(ambient_linear, flash_linear, tau_shadow)
    spec_raw = detect_specularity_mask(flash_linear, saturation_threshold)

    # Per-mask cleanup: open -> close -> conservative dilate.
    shadow = _clean_binary_mask(shadow_raw, radius=morph_radius)
    spec = _clean_binary_mask(spec_raw, radius=morph_radius)

    # Union, then blur for feathering (Section 4.3 final-merge prose).
    merged = np.clip(np.maximum(shadow, spec), 0.0, 1.0)
    merged = _feather_mask(merged, feather_radius)
    return shadow, spec, merged


#-----Provide sRGB linearization------
def srgb_to_linear(image: Array) -> Array:
    """Convert sRGB to linear RGB."""
    image = np.asarray(image, dtype=np.float64)
    image = np.clip(image, 0.0, 1.0)
    low = image / 12.92
    high = ((image + 0.055) / 1.055) ** 2.4
    return np.where(image <= 0.04045, low, high)

def linearize_ambient_to_flash_space(
    ambient_linear_prime: Array,
    iso_ambient: float,
    t_ambient: float,
    iso_flash: float,
    t_flash: float,
) -> Array:
    """
    Implements Eq. (1):
        A_lin = A'_lin * (ISO_F * Delta t_F) / (ISO_A * Delta t_A)
    """
    scale = (iso_flash * t_flash) / (iso_ambient * t_ambient)
    return ambient_linear_prime * scale

#-----Main pipeline steps (Section 4.2 equations)-------

def compute_detail_layer(
    flash: Array,
    sigma_d: float,
    sigma_r: float,
    epsilon: float = 0.02,
    radius: Optional[int] = None,
) -> Array:
    """Compute the flash detail layer.
    Equation mapping
    ----------------
    Implements Eq. (6):
        F_detail = (F + eps) / (F_base + eps)
    """
    flash_base = bilateral_filter(
        flash, sigma_d=sigma_d, sigma_r=sigma_r, radius=radius
    )
    return (flash + epsilon) / (flash_base + epsilon)


def merge_denoised_with_mask(
    ambient_joint: Array,
    ambient_base: Array,
    mask: Array,
) -> Array:
    """
    Implements Eq. (7):
        A_final = (1 - M) * A_NR + M * A_base
    """
    if ambient_joint.shape != ambient_base.shape:
        raise ValueError("ambient_joint and ambient_base must have the same shape")
    
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 2 and ambient_joint.ndim == 3:
        mask = mask[..., None]

    return (1.0 - mask) * ambient_joint + mask * ambient_base

def detail_transfer(
    ambient_joint: Array,
    ambient_base: Array,
    flash_detail: Array,
    mask: Array,
) -> Array:
    """Final image with flash-to-ambient detail transfer.
    Equation mapping
    ----------------
    Implements Eq. (7):
        A_final = (1 - M) * A_NR * F_detail + M * A_base
    """
    if not (ambient_joint.shape == ambient_base.shape == flash_detail.shape):
        raise ValueError(
            "ambient_joint, ambient_base, flash_detail must have matching shapes"
        )

    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 2 and ambient_joint.ndim == 3:
        mask = mask[..., None]
    if mask.shape[:2] != ambient_joint.shape[:2]:
        raise ValueError("mask spatial dimensions must match image")

    return (1.0 - mask) * ambient_joint * flash_detail + mask * ambient_base


def flash_no_flash_pipeline(
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
        ambient_linear = srgb_to_linear(ambient)
        flash_linear = srgb_to_linear(flash)
        scale = ambient_linear.mean() / (flash_linear.mean() + 1e-12)
        ambient_linear *= scale

    shadow_mask, specularity_mask, artifact_mask = detect_flash_artifact_mask(
        ambient_linear=ambient_linear,
        flash_linear=flash_linear,
        tau_shadow=tau_shadow,
        saturation_threshold=saturation_threshold,
        morph_radius=morph_radius,
        feather_radius=feather_radius,
    )

    flash_detail = compute_detail_layer(
        flash,
        sigma_d=sigma_d,
        sigma_r=sigma_r_joint,
        epsilon=0.02,
        radius=radius,
    )
    result = detail_transfer(ambient_joint, ambient_base, flash_detail, artifact_mask)

    return DenoisingResult(
        ambient_base=ambient_base,
        ambient_joint=ambient_joint,
        shadow_mask=shadow_mask,
        specularity_mask=specularity_mask,
        artifact_mask=artifact_mask,
        result=result,
    )


__all__ = [
    "DenoisingResult",
    "srgb_to_linear",
    "linearize_ambient_to_flash_space",
    "detect_shadow_mask",
    "detect_specularity_mask",
    "detect_flash_artifact_mask",
    "compute_detail_layer",
    "detail_transfer",
    "merge_denoised_with_mask",
    "flash_no_flash_pipeline",
]
