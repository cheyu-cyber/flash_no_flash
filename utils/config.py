"""Load flash / no-flash data generation settings from config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class CameraConfig:
    fov_h: float = 60.0  # vertical field of view in degrees
    fov_w: float = 60.0  # horizontal field of view in degrees


@dataclass
class SceneConfig:
    num_shapes_range: Tuple[int, int] = (5, 15)
    shape_types: List[str] = field(default_factory=lambda: ["circle", "rectangle", "triangle", "ellipse"])
    depth_range: Tuple[float, float] = (1.0, 5.0)
    background_depth: Tuple[float, float] = (5.5, 10.0)
    max_shape_fraction: float = 0.4


@dataclass
class FlashConfig:
    flash_power: Tuple[float, float] = (500.0, 1200.0)
    flash_position_x: Tuple[float, float] = (200.0, 312.0)
    flash_position_y: Tuple[float, float] = (200.0, 312.0)
    flash_position_z: Tuple[float, float] = (-0.5, 0.5)
    falloff_exponent: Tuple[float, float] = (1.5, 2.5)
    specular_strength: Tuple[float, float] = (0.3, 0.9)
    specular_shininess: Tuple[float, float] = (20.0, 60.0)
    shadow_softness: Tuple[float, float] = (1.5, 5.0)
    ambient_in_flash: Tuple[float, float] = (0.02, 0.1)
    flash_color_temp: Tuple[float, float] = (5000.0, 6000.0)


@dataclass
class AmbientConfig:
    base_illumination: Tuple[float, float] = (0.10, 0.28)
    illumination_variation: Tuple[float, float] = (0.04, 0.12)
    color_temperature_shift: Tuple[Tuple[float, float], ...] = (
        (0.9, 1.1), (0.85, 1.0), (0.75, 0.9)
    )
    # Room lights
    num_room_lights_range: Tuple[int, int] = (1, 3)
    room_light_power: Tuple[float, float] = (0.3, 1.0)
    room_light_depth: Tuple[float, float] = (0.5, 4.0)
    room_light_falloff: Tuple[float, float] = (1.0, 2.0)
    room_light_softness: float = 30.0
    # Depth fog
    fog_strength: Tuple[float, float] = (0.0, 0.15)
    fog_color: Tuple[float, float, float] = (0.7, 0.75, 0.8)
    # No-flash degradation
    no_flash_darken: Tuple[float, float] = (0.3, 0.7)


@dataclass
class NoiseConfig:
    ambient_noise_std: Tuple[float, float] = (0.02, 0.05)
    flash_noise_std: Tuple[float, float] = (0.004, 0.015)
    poisson_enabled: bool = True
    poisson_peak: Tuple[float, float] = (120.0, 300.0)


@dataclass
class GenerationConfig:
    num_train: int = 100
    num_val: int = 20
    output_dir: str = "./data/synthetic"


@dataclass
class ModelConfig:
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    bottleneck_channels: int = 512
    attention_heads: int = 4
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    loss_l1_weight: float = 1.0
    loss_perceptual_weight: float = 0.1
    loss_gate_reg_weight: float = 0.01
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    val_interval: int = 5
    num_workers: int = 4


@dataclass
class SyntheticDataConfig:
    image_size: Tuple[int, int] = (512, 512)
    seed: int = 1869
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    flash: FlashConfig = field(default_factory=FlashConfig)
    ambient: AmbientConfig = field(default_factory=AmbientConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SyntheticDataConfig:
    """Read config.json and return a SyntheticDataConfig."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cam_raw = data.get("camera", {})
    camera = CameraConfig(
        fov_h=cam_raw.get("fov_h", 60.0),
        fov_w=cam_raw.get("fov_w", 60.0),
    )

    scene_raw = data.get("scene", {})
    flash_raw = data.get("flash", {})
    ambient_raw = data.get("ambient", {})
    noise_raw = data.get("noise", {})
    gen_raw = data.get("generation", {})

    scene = SceneConfig(
        num_shapes_range=tuple(scene_raw.get("num_shapes_range", [5, 15])),
        shape_types=scene_raw.get("shape_types", ["circle", "rectangle", "triangle", "ellipse"]),
        depth_range=tuple(scene_raw.get("depth_range", [1.0, 5.0])),
        background_depth=tuple(scene_raw.get("background_depth", [5.5, 10.0])),
        max_shape_fraction=scene_raw.get("max_shape_fraction", 0.4),
    )

    flash = FlashConfig(
        flash_power=tuple(flash_raw.get("flash_power", [500.0, 1200.0])),
        flash_position_x=tuple(flash_raw.get("flash_position_x", [200.0, 312.0])),
        flash_position_y=tuple(flash_raw.get("flash_position_y", [200.0, 312.0])),
        flash_position_z=tuple(flash_raw.get("flash_position_z", [-0.5, 0.5])),
        falloff_exponent=tuple(flash_raw.get("falloff_exponent", [1.5, 2.5])),
        specular_strength=tuple(flash_raw.get("specular_strength", [0.3, 0.9])),
        specular_shininess=tuple(flash_raw.get("specular_shininess", [20.0, 60.0])),
        shadow_softness=tuple(flash_raw.get("shadow_softness", [1.5, 5.0])),
        ambient_in_flash=tuple(flash_raw.get("ambient_in_flash", [0.02, 0.1])),
        flash_color_temp=tuple(flash_raw.get("flash_color_temp", [5000.0, 6000.0])),
    )

    ct_raw = ambient_raw.get("color_temperature_shift", [[0.9, 1.1], [0.85, 1.0], [0.75, 0.9]])
    ambient = AmbientConfig(
        base_illumination=tuple(ambient_raw.get("base_illumination", [0.10, 0.28])),
        illumination_variation=tuple(ambient_raw.get("illumination_variation", [0.04, 0.12])),
        color_temperature_shift=tuple(tuple(ch) for ch in ct_raw),
        num_room_lights_range=tuple(ambient_raw.get("num_room_lights_range", [1, 3])),
        room_light_power=tuple(ambient_raw.get("room_light_power", [0.3, 1.0])),
        room_light_depth=tuple(ambient_raw.get("room_light_depth", [0.5, 4.0])),
        room_light_falloff=tuple(ambient_raw.get("room_light_falloff", [1.0, 2.0])),
        room_light_softness=ambient_raw.get("room_light_softness", 30.0),
        fog_strength=tuple(ambient_raw.get("fog_strength", [0.0, 0.15])),
        fog_color=tuple(ambient_raw.get("fog_color", [0.7, 0.75, 0.8])),
        no_flash_darken=tuple(ambient_raw.get("no_flash_darken", [0.3, 0.7])),
    )

    noise = NoiseConfig(
        ambient_noise_std=tuple(noise_raw.get("ambient_noise_std", [0.02, 0.05])),
        flash_noise_std=tuple(noise_raw.get("flash_noise_std", [0.004, 0.015])),
        poisson_enabled=noise_raw.get("poisson_enabled", True),
        poisson_peak=tuple(noise_raw.get("poisson_peak", [120.0, 300.0])),
    )

    generation = GenerationConfig(
        num_train=gen_raw.get("num_train", 100),
        num_val=gen_raw.get("num_val", 20),
        output_dir=gen_raw.get("output_dir", "./data/synthetic"),
    )

    model_raw = data.get("model", {})
    model = ModelConfig(
        encoder_channels=tuple(model_raw.get("encoder_channels", [64, 128, 256, 512])),
        bottleneck_channels=model_raw.get("bottleneck_channels", 512),
        attention_heads=model_raw.get("attention_heads", 4),
        decoder_channels=tuple(model_raw.get("decoder_channels", [256, 128, 64, 32])),
        batch_size=model_raw.get("batch_size", 4),
        learning_rate=model_raw.get("learning_rate", 1e-4),
        weight_decay=model_raw.get("weight_decay", 1e-5),
        num_epochs=model_raw.get("num_epochs", 200),
        loss_l1_weight=model_raw.get("loss_l1_weight", 1.0),
        loss_perceptual_weight=model_raw.get("loss_perceptual_weight", 0.1),
        loss_gate_reg_weight=model_raw.get("loss_gate_reg_weight", 0.01),
        checkpoint_dir=model_raw.get("checkpoint_dir", "./checkpoints"),
        log_interval=model_raw.get("log_interval", 10),
        val_interval=model_raw.get("val_interval", 5),
        num_workers=model_raw.get("num_workers", 4),
    )

    return SyntheticDataConfig(
        image_size=tuple(data.get("image_size", [512, 512])),
        seed=data.get("seed", 1869),
        camera=camera,
        scene=scene,
        flash=flash,
        ambient=ambient,
        noise=noise,
        generation=generation,
        model=model,
    )
