"""Assemble the neuro-manipulator with the ada_assets wheelchair and human

Composes three MCJF components into a single MuJoCo scene via MjSpec:
    1. Wheelchair (Permobil frame with arm_attachment_site)
    2. Assistive NeuroRobot Manipulator (attached at arm_attachment_site)
    3. Seated Human (body collision envelope, head mesh, mouth site)

Usage:
    uv run python -m modular_arm.visualization.assemble_scene
    uv run python -m modular_arm.visualization.assemble_scene --arm-xml custom_arm.xml
    uv run python -m modular_arm.visualization.assemble_scene --save-xml scene.xml
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

logger = structlog.get_logger(__name__)

# --- Paths --- TODO: @FRM, move to config file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARM_XML = PROJECT_ROOT / "data" / "robot-models" / "mjcf" / "neuro_arm.xml"

# --- ARM attachment geometry ---
# JACO mounts at [-0.02, 0.02, 0.05] relative to the wheelchair mesh origin (z=0.4612 in the floor frame).
# That position was derived from the ADA planning scene where the wheelchair sits at [0.02, -0.02, -0.05] from the
# JACO root.
#
# The neuro arm has a wider base cylinder (r=0.033 m vs JACO's ~0.020 m) and a different vertical profile.
# The x-offset pushes the base outward to clear the armrest; y centers it on the right side. z controls the height above
# the whelchair mesh origin.
#
# TODO: @FRM, Measure from physical Permobil armrest or CAD. These are initial estimates for visualization; iterate in viewer.
ARM_ATTACHMENT_POS = np.array([-0.04, 0.00, 0.05])
"""Arm base position relative to wheelchair mesh origin [m].
    x: negative = toward the back of the chair
    y: zero = centered on the right armrest rail
    z: positive = above the wheelchair mesh origin
"""

# --- Sternum reference point ---
# All ADL envelopes from adl_envelope_generator.py are normalized to the sternum (0, 0, 0). This site provides the
# sternum position in the MuJoCo world frame so the FK pipeline can transform EE positions into the ADL hull coordinate frame.
#
# Anthropometic basis (50th-percentile seated male):
#   * Seat surface: ~0.46 m (wheelchair_base z in ada_assets)
#   * Sitting height (seat to crown): ~0.91 m
#   * Sternum (suprasternal notch) from seat: ~0.47 m
#   * Sternum (depth from spine): ~ 0.22 m forward
#
# Floor-frame coordinates:
#   x: 0.24 m - forward from wheelchair center (front of chest)
#   y: 0.34 m - body midline (matches head y in seated.xml)
#   z: 0.93 m - seat_z + sternum_from_seat = 0.46 + 0.47
#
# TODO(@FRM): Adjust per participant anthropometrics. These values match 50th percentile male seated dimensions from Dreyfuss/Tilley.
STERNUM_POS = np.array([0.24, 0.34, 1.05])
"""Sternum (suprasternal notch) position in floor frame [m]"""
STERNUM_SITE_RADIUS = 0.015
"""Visual radius for sternum marker [m]"""

def build_composite_spec(
        arm_xml_path: Path,
        *,
        with_human: bool = True,
        with_floor: bool = True,
) -> mujoco.MjSpec:
    """Compose wheelchair + ARM + seated human into a single MjSpec.

    Args:
        arm_xml_path: Path to the ANRM component MJCF.
        with_human: Include seated human model (body envelope, head, mouth).
        with_floor: Include floor plane and lighting.

    Returns:
        A compiled-ready MjSpec containing all three components.
    """
    # --- 1. Wheelchair as the base spec ---
    wheelchair_path = MODELS_DIR / "wheelchair.xml"
    logger.info("loading_wheelchair", path=str(wheelchair_path))
    spec = mujoco.MjSpec.from_file(str(wheelchair_path))
    spec.meshdir = str(ASSETS_DIR)
    # Scene-level settings (matching ada_assets/assembly.py)
    spec.compiler.degree = False # work in radians throughout

    # --- 2. Attach ANRM to the wheelchair mounting site ---
    arm_site = spec.site("arm_attachment_site")
    arm_site.pos = ARM_ATTACHMENT_POS.tolist()
    logger.info(
        "arm_attachment_adjusted",
        pos=ARM_ATTACHMENT_POS.tolist(),
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
        user_body_pos = np.array([0.0, 0.0, 0.4612])
        sternum_relative = STERNUM_POS - user_body_pos
        sternum_site.pos = sternum_relative.tolist()
        sternum_site.size = [STERNUM_SITE_RADIUS, STERNUM_SITE_RADIUS, STERNUM_SITE_RADIUS]
        sternum_site.rgba = [1.0, 0, 1.0, 0.80]
        logger.info(
            "sternum_site_added",
            floor_frame_pos=STERNUM_POS.tolist(),
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

    return spec

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
    parser.add_argument("--arm-xml", type=Path, default=DEFAULT_ARM_XML,
                        help=f"Path to ANRM component MJCF (default{DEFAULT_ARM_XML})",
                        )
    parser.add_argument("--save-xml", type=Path, default=None,
                        help="If set, save the composed MJCF to this path instead of launching the viewer.",
                        )
    parser.add_argument(
        "--no-human", action="store_true",
        help="Exclude seated human model.",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Log site positions for verification (default: True).",
    )
    args = parser.parse_args()

    spec  = build_composite_spec(
        args.arm_xml,
        with_human=not args.no_human,
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
    logger.info("launching_viewer", n_bodies=model.nbody, n_joints=model.njnt)
    mujoco.viewer.launch(model, data)
    return 0

if __name__ == "__main__":
    sys.exit(main())

