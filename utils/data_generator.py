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

from utils.config import SyntheticDataConfig


def kelvin_to_rgb_tint(temp: float) -> np.ndarray:
    """Convert a color temperature (K) to a normalized RGB tint.

    Uses Tanner Helland's approximation for the 1000-40000 K range.
    The result is normalized so the max channel equals 1, giving a
    relative color cast rather than an absolute intensity.
    """
    t = temp / 100.0

    # Red
    if t <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)

    # Green
    if t <= 66:
        g = 99.4708025861 * np.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)

    # Blue
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(t - 10) - 305.0447927307

    rgb = np.array([r, g, b], dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    rgb /= rgb.max()  # normalize so max channel = 1
    return rgb


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

        # Focal lengths in pixels (pinhole camera model)
        fov_h_rad = np.radians(cfg.camera.fov_h)
        fov_w_rad = np.radians(cfg.camera.fov_w)
        self.focal_y = (self.H / 2.0) / np.tan(fov_h_rad / 2.0)
        self.focal_x = (self.W / 2.0) / np.tan(fov_w_rad / 2.0)

        # Per-sample parameters (re-sampled each call to generate())
        self._p = {}

    def _sample_params(self) -> None:
        """Sample concrete parameter values from config ranges for one scene."""
        r = self.rng.uniform
        fcfg = self.cfg.flash
        acfg = self.cfg.ambient
        ncfg = self.cfg.noise

        self._p = {
            "background_depth": r(*self.cfg.scene.background_depth),
            "flash_position": (r(*fcfg.flash_position_x), r(*fcfg.flash_position_y), r(*fcfg.flash_position_z)),
            "flash_power": r(*fcfg.flash_power),
            "falloff_exponent": r(*fcfg.falloff_exponent),
            "specular_strength": r(*fcfg.specular_strength),
            "specular_shininess": r(*fcfg.specular_shininess),
            "shadow_softness": r(*fcfg.shadow_softness),
            "ambient_in_flash": r(*fcfg.ambient_in_flash),
            "flash_color_tint": kelvin_to_rgb_tint(r(*fcfg.flash_color_temp)),
            "base_illumination": r(*acfg.base_illumination),
            "illumination_variation": r(*acfg.illumination_variation),
            "color_temperature_shift": np.array([r(*ch) for ch in acfg.color_temperature_shift]),
            "fog_strength": r(*acfg.fog_strength),
            "no_flash_darken": r(*acfg.no_flash_darken),
            "ambient_noise_std": r(*ncfg.ambient_noise_std),
            "flash_noise_std": r(*ncfg.flash_noise_std),
            "poisson_peak": r(*ncfg.poisson_peak),
        }

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

    # ------------------------------------------------------------------
    # Texture generation
    # ------------------------------------------------------------------

    _TEXTURE_TYPES = ["noise", "stripes", "checkerboard", "gradient", "speckle"]

    def _generate_texture(self, H: int, W: int) -> np.ndarray:
        """Generate a single-channel texture pattern in [0, 1], shape (H, W).

        A random texture type is chosen each call.  The result is meant to be
        blended with a base colour to give spatial variation to a shape or
        the background.
        """
        ttype = self.rng.choice(self._TEXTURE_TYPES)
        
        # Smooth Perlin-like noise: random field + heavy Gaussian blur
        raw = self.rng.standard_normal((H, W))
        sigma = self.rng.uniform(1.5, 50.0)
        tex = gaussian_filter(raw, sigma=sigma)
        lo, hi = tex.min(), tex.max()
        tex = (tex - lo) / (hi - lo + 1e-8)
        if ttype == "noise":
            return tex

        if ttype == "stripes":
            angle = self.rng.uniform(0, np.pi)
            freq = self.rng.uniform(0.02, 0.12)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            phase = xx * np.cos(angle) + yy * np.sin(angle)
            tex = 0.5 + 0.5 * np.sin(2 * np.pi * freq * phase)

        elif ttype == "checkerboard":
            block = self.rng.integers(8, 48)
            yy, xx = np.mgrid[0:H, 0:W]
            tex = ((yy // block) + (xx // block)) % 2
            tex = tex.astype(np.float64)

        elif ttype == "gradient":
            angle = self.rng.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            proj = xx * np.cos(angle) + yy * np.sin(angle)
            lo, hi = proj.min(), proj.max()
            tex = (proj - lo) / (hi - lo + 1e-8)

        else:  # speckle
            density = self.rng.uniform(0.03, 0.20)
            tex = (self.rng.random((H, W)) < density).astype(np.float64)
            tex = gaussian_filter(tex, sigma=1.5)
            lo, hi = tex.min(), tex.max()
            tex = (tex - lo) / (hi - lo + 1e-8)

        return tex

    def _apply_texture(
        self, base_color: np.ndarray, H: int, W: int, strength: float = 0.0
    ) -> np.ndarray:
        """Create a textured colour field (H, W, 3) from a base RGB colour.

        Parameters
        ----------
        base_color : (3,) array in [0, 1]
        strength   : blend factor, 0 = flat colour, 1 = full texture.
                     If 0, a random value in [0.08, 0.35] is chosen.
        """
        if strength <= 0.0:
            strength = float(self.rng.uniform(0.02, 0.50))
        tex = self._generate_texture(H, W)
        tex_color = self.rng.uniform(0.05, 0.95, size=3)
        textured = (
            (1 - strength) * base_color[None, None, :]
            + strength * tex[:, :, None] * tex_color[None, None, :]
        )
        return np.clip(textured, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def build_scene(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a random scene: reflectance (H,W,3), depth (H,W), normals-z (H,W).

        Returns
        -------
        reflectance : (H, W, 3) in [0, 1]
        depth_map   : (H, W) in metres
        surface_cos : (H, W) approximate cosine of surface normal vs camera direction
        """
        H, W = self.H, self.W

        # Background with texture
        bg_color = self.rng.uniform(0.05, 0.6, size=3)
        reflectance = self._apply_texture(bg_color, H, W)
        depth_map = np.full((H, W), self._p["background_depth"], dtype=np.float64)

        # Surface normal z-component (1 = facing camera, <1 = angled away)
        surface_cos = np.ones((H, W), dtype=np.float64)

        shapes = self._random_shapes()
        for shape in shapes:
            mask = np.zeros((H, W), dtype=np.float64)
            mask = self._draw_shape(mask, shape)
            idx = mask > 0.5

            # Paint textured reflectance
            shape_tex = self._apply_texture(shape["color"], H, W)
            reflectance[idx] = shape_tex[idx]
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
        flash_x, flash_y, _ = self._p["flash_position"]
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
        softness = self._p["shadow_softness"]
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
        power = self._p["flash_power"]
        exp = self._p["falloff_exponent"]

        # Distance from each pixel to flash source (all in metres).
        # Convert pixel offsets to metres using focal length and depth.
        fx, fy, fz = self._p["flash_position"]
        focal = (self.focal_x + self.focal_y) / 2.0
        dx = (self.xx - fx) / focal * depth_map  # lateral offset in metres
        dy = (self.yy - fy) / focal * depth_map  # lateral offset in metres
        dz = depth_map - fz                       # depth offset in metres
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dist = np.maximum(dist, 0.1)  # avoid division by zero

        # Inverse-power falloff
        falloff = power / (dist ** exp)

        # Lambertian diffuse: falloff * cos(angle)
        diffuse = falloff * surface_cos * shadow_map

        # Normalise diffuse so maximum is around 1.0
        d_max = diffuse.max()
        if d_max > 0:
            diffuse /= d_max

        # Specular highlights (Blinn-Phong-like)
        # Approximate half-vector dot normal as angular offset from flash axis
        lateral_dist = np.sqrt(dx ** 2 + dy ** 2)
        angular_offset = lateral_dist / np.maximum(dz, 0.1)  # tan(angle) ≈ angle for small angles
        specular_raw = np.exp(-self._p["specular_shininess"] * angular_offset ** 2)
        specular = self._p["specular_strength"] * specular_raw * surface_cos * shadow_map

        return diffuse, specular

    # ------------------------------------------------------------------
    # Ambient illumination model
    # ------------------------------------------------------------------

    def _ambient_illumination(self, depth_map: np.ndarray, surface_cos: np.ndarray) -> np.ndarray:
        """Create a depth-aware ambient light field (H, W, 3).

        Three layers combine to produce realistic ambient illumination:

        1. Base fill — smooth, low-level illumination with spatial noise.
           This represents indirect light bouncing off walls/ceiling.

        2. Room lights — 1-3 point light sources at random (x, y, depth)
           positions, each with its own color and inverse-power falloff.
           A Lambertian cos(θ) term modulates each light based on the
           angle between the light direction and the surface normal,
           so surfaces facing toward a lamp are brighter than those
           angled away.

        3. Depth fog — a small additive haze that increases with distance.
           Simulates indoor atmospheric scattering (dust, moisture) that
           washes out far objects and reduces their contrast.
        """
        H, W = self.H, self.W
        acfg = self.cfg.ambient

        # ----------------------------------------------------------
        # Layer 1: Base fill light (depth-independent)
        # A uniform low illumination with smooth spatial variation,
        # simulating indirect bounced light in a room.
        # ----------------------------------------------------------
        base = np.full((H, W), self._p["base_illumination"], dtype=np.float64)
        variation = self.rng.normal(0, self._p["illumination_variation"], size=(H, W))
        variation = gaussian_filter(variation, sigma=40.0)
        fill = np.clip(base + variation, 0.02, 0.4)

        # Expand to 3 channels (neutral — no colour cast on clean target)
        ambient_light = fill[:, :, None] * np.ones(3)[None, None, :]

        # ----------------------------------------------------------
        # Layer 2: Random room lights (depth-dependent)
        # Each light is placed at a random (x, y) pixel position and
        # a random depth.  The 3D distance from every pixel to the
        # light determines the contribution via inverse-power falloff.
        # This creates soft gradients — nearby surfaces are brighter,
        # giving the ambient image real depth structure.
        # ----------------------------------------------------------
        lo, hi = acfg.num_room_lights_range
        n_lights = int(self.rng.integers(lo, hi + 1))
        focal = (self.focal_x + self.focal_y) / 2.0

        for _ in range(n_lights):
            # Random position for this room light
            lx = self.rng.uniform(0, W)
            ly = self.rng.uniform(0, H)
            lz = self.rng.uniform(*acfg.room_light_depth)

            # Random power and falloff exponent for this light
            power = self.rng.uniform(*acfg.room_light_power)
            falloff_exp = self.rng.uniform(*acfg.room_light_falloff)

            # Random warm/cool color for this light (slight tint)
            light_color = self.rng.uniform(0.8, 1.2, size=3)
            light_color /= light_color.max()  # normalise so max channel = 1

            # 3D distance from every pixel to this light.
            # Convert pixel offsets to the same scale as depth using
            # the focal length, so dx/dy/dz are all in metres.
            dx = (self.xx - lx) / focal
            dy = (self.yy - ly) / focal
            dz = depth_map - lz
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            dist = np.maximum(dist, 0.01)

            # Lambertian cos(θ) — angle between light direction and
            # surface normal.  The surface normal approximation
            # (surface_cos) gives the z-component facing the camera.
            # The light direction z-component at each pixel is dz/dist
            # (how much the light ray points along the depth axis).
            # Combining: surfaces that face toward the light receive
            # more illumination, surfaces angled away receive less.
            # We mix with a base of 0.3 so no surface goes fully dark
            # — real room light bounces off walls and fills in.
            light_cos = np.clip(dz / dist, 0.0, 1.0)
            lambertian = 0.3 + 0.7 * light_cos * surface_cos

            # Inverse-power falloff: closer surfaces get more light
            contribution = power * lambertian / (dist ** falloff_exp)

            # Smooth the contribution to simulate a soft/diffuse lamp
            # rather than a harsh point source
            contribution = gaussian_filter(contribution, sigma=acfg.room_light_softness)

            # Add this light's contribution (coloured)
            ambient_light += contribution[:, :, None] * light_color[None, None, :]

        # ----------------------------------------------------------
        # Layer 3: Depth fog / haze (depth-dependent)
        # Distant objects pick up a small additive color from
        # atmospheric scattering — dust, moisture, haze.  This
        # washes out far objects (lower contrast) and tints them
        # toward a neutral bluish-grey, encoding depth.
        #
        #   fog_amount = fog_strength * (depth / max_depth)
        #   pixel = pixel * (1 - fog_amount) + fog_color * fog_amount
        #
        # We apply this as an additive/blend on the ambient light
        # rather than on the final image so it interacts naturally
        # with reflectance.
        # ----------------------------------------------------------
        fog_strength = self._p["fog_strength"]
        if fog_strength > 0:
            fog_color = np.array(acfg.fog_color, dtype=np.float64)
            depth_max = depth_map.max()
            if depth_max > 0:
                fog_amount = fog_strength * (depth_map / depth_max)
                # Blend: reduce ambient light and add fog color
                ambient_light = (
                    ambient_light * (1.0 - fog_amount[:, :, None])
                    + fog_color[None, None, :] * fog_amount[:, :, None]
                )

        return ambient_light

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian + optional Poisson noise."""
        noisy = image.copy()

        if self.cfg.noise.poisson_enabled:
            # Scale to photon counts, apply Poisson, scale back
            peak = self._p["poisson_peak"]
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
        self._sample_params()

        # 1. Build the underlying scene
        reflectance, depth_map, surface_cos = self.build_scene()

        # 2. Compute shadow map from depth occlusion
        shadow_map = self._cast_shadows(depth_map)

        # 3. Ambient / no-flash image (compute first so flash can reuse it)
        ambient_light = self._ambient_illumination(depth_map, surface_cos)
        no_flash_clean = reflectance * ambient_light
        no_flash_clean = np.clip(no_flash_clean, 0.0, 1.0)

        # 4. Flash image
        diffuse, specular = self._flash_illumination(depth_map, surface_cos, shadow_map)
        flash_tint = self._p["flash_color_tint"][None, None, :]  # (1, 1, 3)
        flash_light = (diffuse[:, :, None] + specular[:, :, None]) * flash_tint
        flash_ambient_contrib = self._p["ambient_in_flash"]
        flash_clean = reflectance * flash_light + flash_ambient_contrib * no_flash_clean * flash_tint
        flash_clean = np.clip(flash_clean + specular[:, :, None] * 0.3 * flash_tint, 0.0, 1.0)

        # 5. Degrade the no-flash input: darken + colour shift + noise.
        #    The clean target stays neutral; the input simulates a real
        #    low-light capture with underexposure and wrong white balance.
        darken = self._p["no_flash_darken"]
        ct = self._p["color_temperature_shift"]
        no_flash_degraded = no_flash_clean * darken * ct[None, None, :]
        no_flash_degraded = np.clip(no_flash_degraded, 0.0, 1.0)

        # 6. Add noise
        flash_noisy = self._add_noise(flash_clean, self._p["flash_noise_std"])
        no_flash_noisy = self._add_noise(no_flash_degraded, self._p["ambient_noise_std"])

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
