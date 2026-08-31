"""Project configuration: per-section accessors over a single sectioned YAML file.

The YAML is read once and cached. Each domain (paths, adl_envelope, robot, scene, optimizer) has its own frozen settings
dataclass and accessor, so a consumer loads and validates only the section it needs; a robot-only script does not fail
on a malformed `optimizer` block. All paths are resolved against the project root (the directory containing config.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# config.py -> core/ -> modular_arm/ -> src/ -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def _load_raw(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and cache the whole YAML document. Section parsing happens downstream."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return yaml.safe_load(config_path.read_text()) or {}

def _section(name: str) -> dict[str, Any]:
    """Return a top-level section from the YAML document with a clear error if absent."""
    data = _load_raw()
    if name not in data:
        raise KeyError(f"Section '{name}' not found in {DEFAULT_CONFIG_PATH}")
    return data[name]

def _resolve(path: str) -> Path:
    """Resolve a config path relative to the project root."""
    return (PROJECT_ROOT / path).resolve()


# --- Paths ---
@dataclass(frozen=True)
class Paths:
    adl_csv: Path
    adl_envelopes_dir: Path
    robot_mjcf_dir: Path

@lru_cache(maxsize=1)
def get_paths() -> Paths:
    s = _section("paths")
    return Paths(
        adl_csv=_resolve(s["adl_csv"]),
        adl_envelopes_dir=_resolve(s["adl_envelopes_dir"]),
        robot_mjcf_dir=_resolve(s["robot_mjcf_dir"]),
    )

# --- ADL Envelope ---
@dataclass(frozen=True)
class ADLSettings:
    default_cohort: str
    trim_quantiles: tuple[float, float]
    zone_mappings: dict[str, list[str]]
    zone_colors: dict[str, list[float]]

@lru_cache(maxsize=1)
def get_adl_settings() -> ADLSettings:
    s = _section("adl_envelope")
    lo, hi = s["trim_quantiles"]
    return ADLSettings(
        default_cohort=s["default_cohort"],
        trim_quantiles=(float(lo), float(hi)),
        zone_mappings=s["zone_mappings"],
        zone_colors=s["zone_colors"],
    )

# --- Robot ---
@dataclass(frozen=True)
class RobotSettings:
    lengths_in: dict[str, float]

@lru_cache(maxsize=1)
def get_robot_settings() -> RobotSettings:
    s = _section("robot")
    return RobotSettings(lengths_in={k: float(v) for k, v in s["lengths_in"].items()})

# --- Scene ---
@dataclass(frozen=True)
class SceneSettings:
    arm_attachment_pos: tuple[float, float, float]
    sternum_pos: tuple[float, float, float]

@lru_cache(maxsize=1)
def get_scene_settings() -> SceneSettings:
    s = _section("scene")
    return SceneSettings(
        arm_attachment_pos=tuple(float(v) for v in s["arm_attachment_pos"]),
        sternum_pos=tuple(float(v) for v in s["sternum_pos"]),
    )

# --- Optimizer ---
@dataclass(frozen=True)
class OptimizerSettings:
    n_joint_samples: int
    reach_eps_m: float
    objective: str

@lru_cache(maxsize=1)
def get_optimizer_settings() -> OptimizerSettings:
    s = _section("optimizer")
    return OptimizerSettings(
        n_joint_samples=int(s["n_joint_samples"]),
        reach_eps_m=float(s["reach_eps_m"]),
        objective=s["objective"],
    )

