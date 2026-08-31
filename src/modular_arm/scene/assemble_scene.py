"""Assemble the neuro-manipulator with the ada_assets wheelchair and human models

Composes three MCJF components into a single MuJoCo scene via MjSpec:
    1. Wheelchair (Permobil frame with arm_attachment_site)
    2. Assistive NeuroRobot Manipulator (attached at arm_attachment_site)
    3. Seated Human (body collision envelope, head mesh, mouth site)

Usage:
    uv run python -m modular_arm.core.scene.assemble_scene
    uv run python -m modular_arm.core.scene.assemble_scene --arm-xml custom_arm.xml
    uv run python -m modular_arm.core.scene.assemble_scene --save-xml scene.xml
    uv run python -m modular_arm.core.scene.assemble_scene --with-adl-hulls
    uv run python -m modular_arm.core.scene.assemble_scene --with-adl-hulls --hull-dir data/custom/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np
import structlog

from ada_assets import ASSETS_DIR, MODELS_DIR
from modular_arm.core.config import get_adl_settings, get_paths, get_scene_settings

logger = structlog.get_logger(__name__)

STERNUM_SITE_RADIUS = 0.015
"""Visual radius for sternum marker [m]"""

# --- ADL Hull Visuallization ---
UNKNOWN_ZONE_COLOR: list[float] = [0.5, 0.5, 0.5, 0.10]
"""Fallback RGBA for hull STLs whose zone name isn't in the config palette"""

def build_composite_spec(
        arm_xml_path: Path,
        *,
        with_human: bool = True,
        with_floor: bool = True,
        adl_hull_dir: Path | None = None,
) -> mujoco.MjSpec:
    """Compose wheelchair + ARM + seated human into a single MjSpec.

    Args:
        arm_xml_path: Path to the ANRM component MJCF.
        with_human: Include seated human model (body envelope, head, mouth).
        with_floor: Include floor plane and lighting.
        adl_hull_dir: If provided, load ADL hull  STL files from this directory and render them as semi-transparent
            visual geoms anchored at the sternum site.

    Returns:
        A compiled-ready MjSpec containing all components.
    """
    scene = get_scene_settings()
    # --- 1. Wheelchair as the base spec ---
    wheelchair_path = MODELS_DIR / "wheelchair.xml"
    logger.info("loading_wheelchair", path=str(wheelchair_path))
    spec = mujoco.MjSpec.from_file(str(wheelchair_path))
    spec.meshdir = str(ASSETS_DIR)
    # Scene-level settings (matching ada_assets/assembly.py)
    spec.compiler.degree = False # work in radians throughout

    # --- 2. Attach ANRM to the wheelchair mounting site ---
    arm_site = spec.site("arm_attachment_site")
    arm_site.pos = list(scene.arm_attachment_pos)
    logger.info(
        "arm_attachment_adjusted",
        pos=arm_site.pos,
        note="override JACO2 position for neuro-arm base geometry",
    )
    logger.info("loading_arm", path=str(arm_xml_path))
    arm_spec = mujoco.MjSpec.from_file(str(arm_xml_path))
    spec.attach(arm_spec, prefix="", site=arm_site)
    logger.info("arm_attached", site="arm_attachment_site")

    # --- 3. Attach the seated human ---
    if with_human:
        human_path = MODELS_DIR / "seated.xml"
        logger.info("loading_human", path=str(human_path))
        human_spec = mujoco.MjSpec.from_file(str(human_path))
        human_spec.meshdir = str(ASSETS_DIR)

        # Add sternum site to human user_body before attaching.
        # user_body is the body collision envelope at z=0.4612.
        # stenum site needs floor-frame coordinates, but since user_body is already at [0, 0, 0.4612], we express the
        # the site position RELATIVE to user_body.
        user_body = human_spec.body("user_body")
        sternum_site = user_body.add_site()
        sternum_site.name = "sternum"
        sternum_pos = np.array(scene.sternum_pos)
        user_body_pos = np.array([0.0, 0.0, 0.4612])
        sternum_relative = sternum_pos - user_body_pos
        sternum_site.pos = sternum_relative.tolist()
        sternum_site.size = [STERNUM_SITE_RADIUS, STERNUM_SITE_RADIUS, STERNUM_SITE_RADIUS]
        sternum_site.rgba = [1.0, 0, 1.0, 0.80]
        logger.info(
            "sternum_site_added",
            floor_frame_pos=sternum_pos.tolist(),
            body_relative_pos=sternum_relative.tolist(),
        )

        human_frame = spec.worldbody.add_frame()
        spec.attach(human_spec, prefix="human/", frame=human_frame)
        logger.info("human_attached")

    # --- 4. Add floor and lighting ---
    if with_floor:
        light = spec.worldbody.add_light()
        light.pos = [0, 0, 3]
        light.dir = [0, 0, -1]
        light.diffuse = [1, 1, 1]

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [3, 3, 0.05]
        floor.rgba = [0.9, 0.9, 0.9, 1]
        floor.contype = 1
        floor.conaffinity = 1

    if adl_hull_dir is not None:
        _attach_adl_hulls(spec, adl_hull_dir)

    return spec

def _attach_adl_hulls(spec: mujoco.MjSpec, hull_dir: Path) -> None:
    """Load ADL hull STLs and attach them as visual geoms at the sternum site.

    Each STL file is expected to follow the naming convention from adl_envelope_generator.export_hulls_as_stl.
    The zone name is extracted to look up the color

    Hulls are in sternum-relative coordinates, so a container body at the sternum site renders them in the correct
    world location.

    Args:
        spec: The MjSpec to modify (must already have a worldbody)
        hull_dir: Directory containing STL hull files
    """
    stl_files = sorted(hull_dir.glob("*.stl"))
    if not stl_files:
        logger.warning("no_adl_envelopes_found", dir=str(hull_dir))
        return

    # Container body at the sternum site - hull vertices are sternum-relative
    hull_body = spec.worldbody.add_body()
    hull_body.name="adl_hull_container"
    hull_body.pos = list(get_scene_settings().sternum_pos)
    zone_colors = get_adl_settings().zone_colors
    for stl_path in stl_files:
        # Extract zone name from filename
        parts = stl_path.stem.rsplit("_", 1)
        zone_name = parts[0] if len(parts) > 1 else stl_path.stem
        color = zone_colors.get(zone_name, UNKNOWN_ZONE_COLOR)

        mesh_name = f"adl_{stl_path.stem}"

        # Register mesh asset
        mesh = spec.add_mesh()
        mesh.name = mesh_name
        mesh.file = str(stl_path)

        # Visual-only geom; no collision or physics
        geom = hull_body.add_geom()
        geom.name = f"{mesh_name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.rgba = color

        logger.info("hull_geom_added", file=stl_path.name, zone=zone_name)

    logger.info("adl_hulls_attached", n_hulls=len(stl_files))

def verify_sites(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Log positions of key sites for visual inspection and verification.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data after mj_forward().
    """
    mujoco.mj_forward(model, data)

    site_names = ["arm_attachment_site", "ee_site", "human/sternum", "human/mouth"]

    for name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            pos = data.site_xpos[site_id].copy()
            logger.info("site_position", name=name, pos=pos.tolist())
        else:
            logger.warning("site_not_found", name=name)
    # Compute EE-to-sternum and EE-to-mouth distances at home config
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    sternum_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "human/sternum")
    mouth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "human/mouth")

    if ee_id >= 0 and sternum_id >= 0:
        dist = np.linalg.norm(data.site_xpos[ee_id] - data.site_xpos[sternum_id])
        logger.info("ee_to_sternum_distance", meters=f"{dist:.3f}")
    if ee_id >= 0 and mouth_id >= 0:
        dist = np.linalg.norm(data.site_xpos[ee_id] - data.site_xpos[mouth_id])
        logger.info("ee_to_mouth_distance", meters=f"{dist:.3f}")

def main() -> int:
    """CLI entrypoint: assemble and launch the composite scene."""
    parser = argparse.ArgumentParser(description="Assemble Assistive NeuroRobot Manipulator + Wheelchair + Human scene.")
    parser.add_argument("--arm-xml", type=Path, default=None,
                        help=f"Path to ANRM component MJCF (default: paths.robot_mjcf_dir/neuro_arm.xml)",
                        )
    parser.add_argument("--save-xml", type=Path, default=None,
                        help="If set, save the composed MJCF to this path instead of launching the viewer.",
                        )
    parser.add_argument(
        "--no-human", action="store_true",
        help="Exclude seated human model.",
    )
    parser.add_argument(
        "--with-adl-hulls", action="store_true",
        help="Load ADL hull STLs and render them at the sternum position"
    )
    parser.add_argument(
        "--hull-dir", type=Path, default=None,
        help=f"Directory containing ADL hull STLs (default: paths.adl_envelopes_dir from config)",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Log site positions for verification (default: True).",
    )
    args = parser.parse_args()

    arm_xml = args.arm_xml or (get_paths().robot_mjcf_dir / "neuro_arm.xml")
    hull_dir = args.hull_dir or get_paths().adl_envelopes_dir
    spec  = build_composite_spec(
        arm_xml,
        with_human=not args.no_human,
        adl_hull_dir=hull_dir if args.with_adl_hulls else None,
    )

    if args.save_xml is not None:
        xml_string = spec.to_xml()
        args.save_xml.write_text(xml_string)
        logger.info("composite_model_saved", path=str(args.save_xml))
        return 0

    # Compile and launch interactive viewer
    model = spec.compile()
    data = mujoco.MjData(model)
    # Apply keyframe if it exists
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    if args.verify:
        verify_sites(model, data)
    logger.info(
        "launching_viewer",
        n_bodies=model.nbody, n_joints=model.njnt, n_actuators=model.nu,
    )
    mujoco.viewer.launch(model, data)
    return 0

if __name__ == "__main__":
    sys.exit(main())

