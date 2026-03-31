"""Load flash / no-flash data generation settings from config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class SceneConfig:
    num_shapes_range: Tuple[int, int] = (5, 15)
    shape_types: List[str] = field(default_factory=lambda: ["circle", "rectangle", "triangle", "ellipse"])
    depth_range: Tuple[float, float] = (1.0, 5.0)
    background_depth: float = 6.0
    max_shape_fraction: float = 0.4


@dataclass
class FlashConfig:
    flash_power: float = 800.0
    flash_position: Tuple[float, float, float] = (256.0, 256.0, 0.0)
    falloff_exponent: float = 2.0
    specular_strength: float = 0.6
    specular_shininess: float = 40.0
    shadow_softness: float = 3.0
    ambient_in_flash: float = 0.05


@dataclass
class AmbientConfig:
    base_illumination: float = 0.18
    illumination_variation: float = 0.08
    color_temperature_shift: Tuple[float, float, float] = (1.0, 0.95, 0.85)


@dataclass
class NoiseConfig:
    ambient_noise_std: float = 0.035
    flash_noise_std: float = 0.008
    poisson_enabled: bool = True
    poisson_peak: float = 200.0


@dataclass
class GenerationConfig:
    num_train: int = 100
    num_val: int = 20
    output_dir: str = "./data/synthetic"


@dataclass
class ModelConfig:
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    depth_encoder_channels: Tuple[int, ...] = (16, 32, 64, 128)
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

    scene_raw = data.get("scene", {})
    flash_raw = data.get("flash", {})
    ambient_raw = data.get("ambient", {})
    noise_raw = data.get("noise", {})
    gen_raw = data.get("generation", {})

    scene = SceneConfig(
        num_shapes_range=tuple(scene_raw.get("num_shapes_range", [5, 15])),
        shape_types=scene_raw.get("shape_types", ["circle", "rectangle", "triangle", "ellipse"]),
        depth_range=tuple(scene_raw.get("depth_range", [1.0, 5.0])),
        background_depth=scene_raw.get("background_depth", 6.0),
        max_shape_fraction=scene_raw.get("max_shape_fraction", 0.4),
    )

    flash = FlashConfig(
        flash_power=flash_raw.get("flash_power", 800.0),
        flash_position=tuple(flash_raw.get("flash_position", [256.0, 256.0, 0.0])),
        falloff_exponent=flash_raw.get("falloff_exponent", 2.0),
        specular_strength=flash_raw.get("specular_strength", 0.6),
        specular_shininess=flash_raw.get("specular_shininess", 40.0),
        shadow_softness=flash_raw.get("shadow_softness", 3.0),
        ambient_in_flash=flash_raw.get("ambient_in_flash", 0.05),
    )

    ambient = AmbientConfig(
        base_illumination=ambient_raw.get("base_illumination", 0.18),
        illumination_variation=ambient_raw.get("illumination_variation", 0.08),
        color_temperature_shift=tuple(ambient_raw.get("color_temperature_shift", [1.0, 0.95, 0.85])),
    )

    noise = NoiseConfig(
        ambient_noise_std=noise_raw.get("ambient_noise_std", 0.035),
        flash_noise_std=noise_raw.get("flash_noise_std", 0.008),
        poisson_enabled=noise_raw.get("poisson_enabled", True),
        poisson_peak=noise_raw.get("poisson_peak", 200.0),
    )

    generation = GenerationConfig(
        num_train=gen_raw.get("num_train", 100),
        num_val=gen_raw.get("num_val", 20),
        output_dir=gen_raw.get("output_dir", "./data/synthetic"),
    )

    model_raw = data.get("model", {})
    model = ModelConfig(
        encoder_channels=tuple(model_raw.get("encoder_channels", [64, 128, 256, 512])),
        depth_encoder_channels=tuple(model_raw.get("depth_encoder_channels", [16, 32, 64, 128])),
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
        scene=scene,
        flash=flash,
        ambient=ambient,
        noise=noise,
        generation=generation,
        model=model,
    )
