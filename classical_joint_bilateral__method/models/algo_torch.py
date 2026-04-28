"""Vectorized PyTorch versions of the bilateral / joint bilateral filters.

Drop-in replacements for `bilateral_filter` and `joint_bilateral_filter` in
`algo.py`. Same call signature, same numpy-in / numpy-out contract — so call
sites that previously did

    out = bilateral_filter(image, sigma_d, sigma_r, radius=r)

can switch to

    out = bilateral_filter_torch(image, sigma_d, sigma_r, radius=r)

without other changes.

The math follows Eqs. (2)-(4) of Petschnigg et al. (2004); the only difference
from the reference numpy loop is that we vectorize over the spatial window with
`F.unfold` (im2col) so the inner double loop becomes a single batched op.
Because `g_r(x) = exp(-x^2 / 2σ_r^2)` only uses the squared distance, we avoid
the explicit sqrt in `np.linalg.norm` and operate on squared L2 directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

Array = np.ndarray


def _default_radius(sigma_d: float) -> int:
    return max(1, int(np.ceil(3.0 * sigma_d)))


def _spatial_kernel_flat(radius: int, sigma_d: float, *, device, dtype) -> torch.Tensor:
    """Return the flattened (k*k,) 2D Gaussian spatial kernel."""
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = xs * xs + ys * ys
    kernel = torch.exp(-dist2 / (2.0 * sigma_d * sigma_d))
    return kernel.reshape(-1)  # (k*k,)


def _to_nchw(image: Array, *, device, dtype) -> torch.Tensor:
    """HxWxC numpy -> (1, C, H, W) torch on `device`."""
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {image.shape}")
    t = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=dtype)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def _from_nchw(tensor: torch.Tensor) -> Array:
    """(1, C, H, W) torch -> HxWxC numpy float64 (matches algo.py output dtype)."""
    return tensor.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float64)


def bilateral_filter_torch(
    image: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> Array:
    """Vectorized bilateral filter (Eqs. 2-3) on GPU/CPU via torch.

    Numerically equivalent to `algo.bilateral_filter` up to floating point
    rounding. Default dtype is float64 to match the reference; pass
    `dtype=torch.float32` for ~2x speedup on GPU at minor precision cost.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    radius = _default_radius(sigma_d) if radius is None else int(radius)
    k = 2 * radius + 1

    img = _to_nchw(image, device=device, dtype=dtype)              # (1, C, H, W)
    _, C, H, W = img.shape

    # Reflect-pad to match np.pad(..., mode="reflect").
    padded = F.pad(img, (radius, radius, radius, radius), mode="reflect")

    # Im2col: (1, C*k*k, H*W) -> (1, C, k*k, H, W)
    patches = F.unfold(padded, kernel_size=k).view(1, C, k * k, H, W)

    # Center A_p, broadcast across the k*k window dim.
    center = img.unsqueeze(2)                                      # (1, C, 1, H, W)
    diff = patches - center                                        # (1, C, k*k, H, W)

    # |A_{p'} - A_p|^2  (sum over channel dim = squared L2 across channels)
    dist2 = diff.pow(2).sum(dim=1)                                 # (1, k*k, H, W)

    # g_r — operate on squared distance, no sqrt needed.
    range_w = torch.exp(-dist2 / (2.0 * sigma_r * sigma_r))        # (1, k*k, H, W)

    # g_d * g_r — spatial kernel broadcasts over (1, k*k, H, W).
    spatial = _spatial_kernel_flat(radius, sigma_d, device=device, dtype=dtype)
    weights = range_w * spatial.view(1, k * k, 1, 1)               # (1, k*k, H, W)

    # k(p) and weighted sum over the window.
    norm = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)       # (1, 1, H, W)
    numer = (weights.unsqueeze(1) * patches).sum(dim=2)            # (1, C, H, W)
    out = numer / norm                                             # (1, C, H, W)

    return _from_nchw(out)


def joint_bilateral_filter_torch(
    ambient: Array,
    flash: Array,
    sigma_d: float,
    sigma_r: float,
    radius: Optional[int] = None,
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> Array:
    """Vectorized joint bilateral filter (Eq. 4) on GPU/CPU via torch.

    The range weight is computed from the flash image F; the ambient image A is
    averaged. Numerically equivalent to `algo.joint_bilateral_filter` up to
    floating point rounding.
    """
    if ambient.shape != flash.shape:
        raise ValueError("ambient and flash must have the same shape")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    radius = _default_radius(sigma_d) if radius is None else int(radius)
    k = 2 * radius + 1

    a_img = _to_nchw(ambient, device=device, dtype=dtype)
    f_img = _to_nchw(flash, device=device, dtype=dtype)
    _, C, H, W = a_img.shape

    a_pad = F.pad(a_img, (radius, radius, radius, radius), mode="reflect")
    f_pad = F.pad(f_img, (radius, radius, radius, radius), mode="reflect")

    a_patches = F.unfold(a_pad, kernel_size=k).view(1, C, k * k, H, W)
    f_patches = F.unfold(f_pad, kernel_size=k).view(1, C, k * k, H, W)

    f_center = f_img.unsqueeze(2)                                  # (1, C, 1, H, W)
    f_diff = f_patches - f_center
    dist2 = f_diff.pow(2).sum(dim=1)                               # (1, k*k, H, W)

    range_w = torch.exp(-dist2 / (2.0 * sigma_r * sigma_r))
    spatial = _spatial_kernel_flat(radius, sigma_d, device=device, dtype=dtype)
    weights = range_w * spatial.view(1, k * k, 1, 1)

    norm = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    numer = (weights.unsqueeze(1) * a_patches).sum(dim=2)
    out = numer / norm

    return _from_nchw(out)


__all__ = [
    "bilateral_filter_torch",
    "joint_bilateral_filter_torch",
]
