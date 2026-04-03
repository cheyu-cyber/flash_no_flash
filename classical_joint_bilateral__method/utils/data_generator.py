"""Core flash / no-flash synthetic data generation.

Generates paired (flash, no-flash) images with physically-motivated depth-based
lighting: inverse-square flash falloff, shadow casting, specular highlights,
and harsh contrast typical of on-camera flash photography.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from config import SyntheticDataConfig


@dataclass
class SceneSample:
    """All outputs for a single generated scene."""
    scene: np.ndarray        # (H, W, 3) underlying reflectance [0, 1]
    depth: np.ndarray        # (H, W)    depth map in metres
    flash: np.ndarray        # (H, W, 3) flash image [0, 1]
    no_flash: np.ndarray     # (H, W, 3) ambient / no-flash image [0, 1]
    flash_clean: np.ndarray  # (H, W, 3) flash without noise [0, 1]
    no_flash_clean: np.ndarray  # (H, W, 3) ambient without noise [0, 1]
    shadow_map: np.ndarray   # (H, W)    shadow intensity [0, 1]
    specular_map: np.ndarray # (H, W)    specular highlight intensity [0, 1]


class FlashNoFlashGenerator:
    """Generates depth-aware flash / no-flash image pairs at 512x512."""

    def __init__(self, cfg: SyntheticDataConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.H, self.W = cfg.image_size
        self.rng = rng or np.random.default_rng(cfg.seed)

        # Pre-compute pixel coordinate grids
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _random_shapes(self) -> List[Dict]:
        """Generate random shape descriptors sorted far-to-near (painter's order)."""
        lo, hi = self.cfg.scene.num_shapes_range
        n = self.rng.integers(lo, hi + 1)
        z_min, z_max = self.cfg.scene.depth_range
        max_dim = int(max(self.H, self.W) * self.cfg.scene.max_shape_fraction)

        shapes = []
        types = self.cfg.scene.shape_types
        for _ in range(n):
            stype = self.rng.choice(types)
            depth = self.rng.uniform(z_min, z_max)
            color = self.rng.uniform(0.08, 0.95, size=3)
            cx = self.rng.integers(0, self.W)
            cy = self.rng.integers(0, self.H)
            shapes.append({
                "type": stype,
                "depth": float(depth),
                "color": color,
                "cx": int(cx),
                "cy": int(cy),
                "max_dim": max_dim,
            })

        # Sort far to near so nearer objects paint on top
        shapes.sort(key=lambda s: -s["depth"])
        return shapes

    def _draw_shape(self, mask: np.ndarray, shape: Dict) -> np.ndarray:
        """Draw a single shape into a binary mask and return it."""
        stype = shape["type"]
        cx, cy = shape["cx"], shape["cy"]
        md = shape["max_dim"]

        if stype == "circle":
            r = self.rng.integers(md // 6, md // 2 + 1)
            cv2.circle(mask, (cx, cy), int(r), 1.0, -1)

        elif stype == "ellipse":
            ax1 = self.rng.integers(md // 6, md // 2 + 1)
            ax2 = self.rng.integers(md // 6, md // 2 + 1)
            angle = self.rng.uniform(0, 180)
            cv2.ellipse(mask, (cx, cy), (int(ax1), int(ax2)), angle, 0, 360, 1.0, -1)

        elif stype == "rectangle":
            hw = self.rng.integers(md // 6, md // 2 + 1)
            hh = self.rng.integers(md // 6, md // 2 + 1)
            angle = self.rng.uniform(0, 180)
            rect = ((cx, cy), (int(hw * 2), int(hh * 2)), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(mask, [box], 0, 1.0, -1)

        elif stype == "triangle":
            size = self.rng.integers(md // 4, md // 2 + 1)
            angles = self.rng.uniform(0, 2 * np.pi, size=3)
            vx = cx + (size * np.cos(angles)).astype(np.int32)
            vy = cy + (size * np.sin(angles)).astype(np.int32)
            pts = np.stack([vx, vy], axis=1).reshape(-1, 1, 2)
            cv2.drawContours(mask, [pts], 0, 1.0, -1)

        return mask

    def build_scene(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a random scene: reflectance (H,W,3), depth (H,W), normals-z (H,W).

        Returns
        -------
        reflectance : (H, W, 3) in [0, 1]
        depth_map   : (H, W) in metres
        surface_cos : (H, W) approximate cosine of surface normal vs camera direction
        """
        H, W = self.H, self.W

        # Background
        bg_color = self.rng.uniform(0.05, 0.6, size=3)
        reflectance = np.ones((H, W, 3), dtype=np.float64) * bg_color
        depth_map = np.full((H, W), self.cfg.scene.background_depth, dtype=np.float64)

        # Surface normal z-component (1 = facing camera, <1 = angled away)
        surface_cos = np.ones((H, W), dtype=np.float64)

        shapes = self._random_shapes()
        for shape in shapes:
            mask = np.zeros((H, W), dtype=np.float64)
            mask = self._draw_shape(mask, shape)
            idx = mask > 0.5

            # Paint reflectance
            reflectance[idx] = shape["color"]
            depth_map[idx] = shape["depth"]

            # Give shapes a slight surface normal variation (dome-like)
            if np.any(idx):
                ys, xs = np.where(idx)
                cy_s, cx_s = ys.mean(), xs.mean()
                max_r = max(np.sqrt((ys - cy_s) ** 2 + (xs - cx_s) ** 2).max(), 1.0)
                r_norm = np.sqrt((ys - cy_s) ** 2 + (xs - cx_s) ** 2) / max_r
                cos_vals = np.sqrt(np.clip(1.0 - 0.3 * r_norm ** 2, 0.3, 1.0))
                surface_cos[ys, xs] = cos_vals

        return reflectance, depth_map, surface_cos

    # ------------------------------------------------------------------
    # Shadow casting
    # ------------------------------------------------------------------

    def _cast_shadows(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute a soft shadow map based on depth occlusion from flash position.

        Objects closer to the camera (smaller depth) cast shadows onto
        regions behind them.  The shadow map is in [0, 1] where 0 = fully
        shadowed and 1 = fully lit.
        """
        H, W = self.H, self.W
        flash_x, flash_y, _ = self.cfg.flash.flash_position
        shadow = np.ones((H, W), dtype=np.float64)

        # For each pixel, trace a ray back toward the flash and check if
        # any closer object blocks the path.  We approximate this by
        # checking a discrete set of points along the ray.
        n_steps = 32
        for step in range(1, n_steps + 1):
            t = step / n_steps  # 0 → 1 (pixel → flash)
            # Intermediate sample positions along ray toward flash center
            sx = (self.xx * (1 - t) + flash_x * t).astype(np.int32).clip(0, W - 1)
            sy = (self.yy * (1 - t) + flash_y * t).astype(np.int32).clip(0, H - 1)
            blocker_depth = depth_map[sy, sx]
            # If the intermediate point is closer than the current pixel, it blocks
            occluded = (blocker_depth < depth_map - 0.15)
            shadow[occluded] *= 0.85  # Accumulate partial occlusion

        # Soften shadow edges
        softness = self.cfg.flash.shadow_softness
        if softness > 0:
            shadow = gaussian_filter(shadow, sigma=softness)

        return np.clip(shadow, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Flash illumination model
    # ------------------------------------------------------------------

    def _flash_illumination(
        self, depth_map: np.ndarray, surface_cos: np.ndarray, shadow_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute flash illumination per pixel using inverse-square falloff.

        Returns
        -------
        diffuse  : (H, W) diffuse flash contribution
        specular : (H, W) specular highlight map
        """
        fcfg = self.cfg.flash
        power = fcfg.flash_power
        exp = fcfg.falloff_exponent

        # Distance from each pixel to flash source (in image-space + depth)
        fx, fy, fz = fcfg.flash_position
        dx = self.xx - fx
        dy = self.yy - fy
        dz = depth_map - fz
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dist = np.maximum(dist, 1.0)  # avoid division by zero

        # Inverse-power falloff
        falloff = power / (dist ** exp)

        # Lambertian diffuse: falloff * cos(angle)
        diffuse = falloff * surface_cos * shadow_map

        # Normalise diffuse so maximum is around 1.0
        d_max = diffuse.max()
        if d_max > 0:
            diffuse /= d_max

        # Specular highlights (Blinn-Phong-like)
        # Approximate half-vector dot normal as function of distance to flash center
        center_dist = np.sqrt(dx ** 2 + dy ** 2)
        center_dist_norm = center_dist / max(self.W, self.H)
        specular_raw = np.exp(-fcfg.specular_shininess * center_dist_norm ** 2)
        specular = fcfg.specular_strength * specular_raw * surface_cos * shadow_map

        return diffuse, specular

    # ------------------------------------------------------------------
    # Ambient illumination model
    # ------------------------------------------------------------------

    def _ambient_illumination(self) -> np.ndarray:
        """Create a smooth, low-level ambient light field (H, W, 3).

        Simulates soft environmental lighting with slight spatial variation
        and warm/cool colour shift.
        """
        acfg = self.cfg.ambient
        H, W = self.H, self.W

        # Smooth random illumination field
        base = np.full((H, W), acfg.base_illumination, dtype=np.float64)
        variation = self.rng.normal(0, acfg.illumination_variation, size=(H, W))
        variation = gaussian_filter(variation, sigma=40.0)
        illum = np.clip(base + variation, 0.02, 0.4)

        # Apply colour temperature shift
        ct = np.array(acfg.color_temperature_shift, dtype=np.float64)
        ambient_light = illum[:, :, None] * ct[None, None, :]

        return ambient_light

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian + optional Poisson noise."""
        ncfg = self.cfg.noise
        noisy = image.copy()

        if ncfg.poisson_enabled:
            # Scale to photon counts, apply Poisson, scale back
            peak = ncfg.poisson_peak
            photons = noisy * peak
            photons = self.rng.poisson(np.clip(photons, 0, peak * 10)).astype(np.float64)
            noisy = photons / peak

        # Additive Gaussian read noise
        noisy += self.rng.normal(0, std, size=noisy.shape)
        return np.clip(noisy, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Full sample generation
    # ------------------------------------------------------------------

    def generate(self) -> SceneSample:
        """Generate one flash / no-flash pair with depth map."""

        # 1. Build the underlying scene
        reflectance, depth_map, surface_cos = self.build_scene()

        # 2. Compute shadow map from depth occlusion
        shadow_map = self._cast_shadows(depth_map)

        # 3. Flash image
        diffuse, specular = self._flash_illumination(depth_map, surface_cos, shadow_map)
        flash_light = diffuse[:, :, None] + specular[:, :, None]
        flash_ambient_contrib = self.cfg.flash.ambient_in_flash
        flash_clean = reflectance * flash_light + flash_ambient_contrib * reflectance
        flash_clean = np.clip(flash_clean + specular[:, :, None] * 0.3, 0.0, 1.0)

        # 4. Ambient / no-flash image
        ambient_light = self._ambient_illumination()
        no_flash_clean = reflectance * ambient_light
        no_flash_clean = np.clip(no_flash_clean, 0.0, 1.0)

        # 5. Add noise
        flash_noisy = self._add_noise(flash_clean, self.cfg.noise.flash_noise_std)
        no_flash_noisy = self._add_noise(no_flash_clean, self.cfg.noise.ambient_noise_std)

        return SceneSample(
            scene=reflectance,
            depth=depth_map,
            flash=flash_noisy,
            no_flash=no_flash_noisy,
            flash_clean=flash_clean,
            no_flash_clean=no_flash_clean,
            shadow_map=shadow_map,
            specular_map=specular,
        )
