"""Procedural MuJoCo model generator for the Assistive Neuro Robot Manipulator (ANRM).

Two output modes:
* ``generate_mjcf`` (full standalone scene)
* ``generate_component_mjcf`` (composable component for ada_assets wheelchair integration via ``MjSpec.attach()``).
"""

from __future__ import  annotations

import argparse
import sys
from typing import List
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt
import structlog

from modular_arm.robot.sea_module import SEAModule

logger = structlog.get_logger(__name__)

# --- Rendering Constants ---
# Visual / collision geometry sizing.
# Physical params (link lengths, masses, spring rates) live in robot_config.py -> config.yaml

# TODO: @FRM, move to config file
LINK_CAPSULE_RADIUS_M: float = 0.025 # Standard arm link capsule [m]
BASE_CAPSULE_RADIUS_M: float = 0.033 # Base cylinder (DOF1) capsule [m]
EE_SITE_RADIUS_M: float      = 0.010 # End-effector site marker [m]
JOINT_ARMATURE: float        = 0.100 # Rotator inertia, revolute joints TODO: @FRM, estimate from hardware
PRISMATIC_ARMATURE: float    = 0.010 # Rotator inertia, prismatic joints

# Placeholder inertia. TODO: @FRM, compute from link geometry
DEFAULT_DIAGINERTIA: str = "0.010 0.010 0.010"

# Material defs for component output (Menagerie visual style)
MATERIALS: dict[str, dict[str, str]] = {
    "arm_link":{
        "rgba": "0.5 0.5 0.5 1",
        "specular": "0.3",
        "shininess": "0.2",
    },
    "arm_base":{
        "rgba": "0.250 0.250 0.250 1",
        "specular": "0.2",
        "shininess": "0.15",
    },
    "arm_joint_ring":{
        "rgba": "0.700 0.700 0.700 1",
        "specular": "0.5",
        "shininess": "0.4",
    },
}

class ArmAssembler:
    """
    Procedurally generate MuJoCo models from a list of SEA modules.

    Supports two output modes:
    1. Full standalone scene (generate_mjcf)
    2. Composable component (generate_component_mjcf) that follows the ada_assets
        attachment-site pattern for wheelchair integration.
    """

    def __init__(self, base_position: npt.NDArray[np.floating] ) -> None:
        self.base_position = base_position
        self.modules: List[SEAModule] = []
        self.link_lengths: List[float] = []

    def add_module(self, module: SEAModule, link_length: float) -> None:
        """
        Appends a module with a specific link length to the kinematic chain.

        Args:
            module: SEA module defining joint type, axis, limits, and dynamics
            link_length: Structural length of this module's link [m]
        """
        self.modules.append(module)
        self.link_lengths.append(link_length)

    # --- Private Helper Methods ---

    def _array_to_str(self, arr: npt.NDArray) -> str:
        """
        Helper function to convert a numpy array to space-separated string for MJCF.
        """
        return " ".join(map(str, arr))

    def _is_base_link(self, module:SEAModule) -> bool:
        """
        Returns True if the module is the base cylinder (wider capsule)
        """
        return "dof1" in module.name

    def _format_range(self, module: SEAModule, *, use_radians:bool = False) -> str | None:
        """
        Format joint range as a string, converting units if necessary.

        Args:
            module: SEA module whose range is to be formatted.
            use_radians: If True, keep radians, otherwise convert to degrees.

        Returns:
            Formatted range string, or None if no range is defined.
        """
        if not hasattr(module, "joint_range") or module.joint_range is None:
            return None
        if module.is_prismatic or use_radians:
            low, high = module.joint_range[0], module.joint_range[1]
        else:
            low = np.rad2deg(module.joint_range[0])
            high = np.rad2deg(module.joint_range[1])

        return f"{low:.6f} {high:.6f}"

    def _build_link_geom_attrs(self, module: SEAModule, link_length:float) -> dict[str, str] | None:
        """
        Compute the from-to and size attributes for a link capsule.

        Args:
            module: SEA module defining link direction.
            link_length: Structural length of this module's link [m]

        Returns:
            Dict with "fromto" and "size" keys, or None if link_length is 0 (pure twist joint with no structural segment).
        """
        if link_length <= 0:
            return None

        dir_vec = getattr(module, "link_direction", np.array([1, 0, 0]))
        end_point = dir_vec * link_length
        return {
            "fromto": f"0 0 0 {end_point[0]} {end_point[1]} {end_point[2]}",
            "size": str( BASE_CAPSULE_RADIUS_M if self._is_base_link(module) else LINK_CAPSULE_RADIUS_M),
        }

    # --- Full Scene Output ---

    def generate_mjcf(self) -> str:
        """Build a complete MuJoCo XML with floor, lighting, and sim options.

        Iterate through self.modules, creating <body>, <joint>, and <geom>, tags for each SEA module in the chain.

        Returns:
            Pretty-printed MJCF XML string.
        """
        # --- Create root and defaults ---
        root = ET.Element("mujoco", model="neuro_manipulator")

        # --- Global settings ---
        ET.SubElement(root, "compiler", angle="degree", coordinate="local", autolimits="true")
        option = ET.SubElement(root, "option", integrator="RK4", timestep="0.0005")
        ET.SubElement(option, "flag", energy="enable")

        # --- Assets and Visuals ---
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                      rgb1=".1 .2 .3", rgb2=".2 .3 .4", width="300", height="300")
        ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="1 1")

        # --- Worldbody ---
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1", diffuse="1 1 1")
        ET.SubElement(worldbody, "geom", name="floor", type="plane", size="2 2 0.01", material="grid")

        # anchor body stays fixed to the world (no joint). it is our world-fixed attachment point
        last_parent = ET.SubElement(worldbody, "body", name="anchor", pos=self._array_to_str(self.base_position))

        # Initialize actuator list
        actuators: list[str] = []

        # Start recursive chain at base position
        for i, (module, link_length) in enumerate(zip(self.modules, self.link_lengths)):
            # 1. create module body
            # if first module, pos is 0 0 0 wrt to anchor
            # otherwise, it's set by previous iter attach_pos
            current_body = ET.SubElement(last_parent, "body", name=f"body_{module.name}", pos="0 0 0")

            # 2. add joint
            joint_type = "slide" if module.is_prismatic else "hinge"

            # Pull limits from module if they exist, default to none
            joint_kwargs = {
                "name": f"{module.name}_joint",
                "type": joint_type,
                "axis": self._array_to_str(module.axis),
                "stiffness": str(module.k_s),
                "damping": str(module.c_s),
            }
            # Apply range if defined based on joint type
            range_str = self._format_range(module, use_radians=False)
            if range_str is not None:
                joint_kwargs["range"] = range_str
            ET.SubElement(current_body, "joint", **joint_kwargs)

            # 3. define Inertial Physics
            # Add small diaginertia to prevent mjMINVAL error in the real model, this would be calculated from geometry
            ET.SubElement(current_body, "inertial",
                          pos=self._array_to_str(module.com),
                          mass=str(module.m),
                          diaginertia=DEFAULT_DIAGINERTIA) # TODO:@FRM figure out how to estimate inertia

            # 4. define Link Geometry
            # Dynamically project capsule geometry based on custom link directions
            # Only generate a capsule mesh if the link has a physical structural length
            geom_attrs = self._build_link_geom_attrs(module, link_length)
            if geom_attrs is not None:
                ET.SubElement(
                    current_body, "geom",
                    name=f"{module.name}_link ",
                    type="capsule",
                    fromto=geom_attrs["fromto"],
                    size=geom_attrs["size"],
                    rgba="0.5 0.5 0.5 1")

            # Store actuator info if applicable
            if module.is_actuated:
                actuators.append(module.name)

            # # 5. Prepare for next iter. create a child body for the NEXT module at the end of THIS link
            # if i < len(self.modules)-1:
            #     # use attachment point in module if available
            #     attach_pos = module.r_attach if np.any(module.r_attach) else np.array([0, 0, link_length])
            #     last_parent = ET.SubElement(current_body, "body",
            #                                    name = f"mount_{i}",
            #                                    pos=self._array_to_str(attach_pos))
            # else:
            #     # At end-effector at the tip of the last link
            #     ET.SubElement(current_body, "site", name="ee_site", pos=f"0 0 {link_length}",  size="0.01")

            # 5. Prepare for next iter
            if i < len(self.modules) - 1:
                # Change default attachment from Z to X
                attach_pos = module.r_attach if np.any(module.r_attach) else np.array([link_length, 0, 0])
                last_parent = ET.SubElement(current_body, "body",
                                            name=f"mount_{i}",
                                            pos=self._array_to_str(attach_pos))
            else:
                # Update end-effector site to the tip on the X-axis
                ET.SubElement(current_body, "site", name="ee_site", pos=f"{link_length} 0 0", size=str(EE_SITE_RADIUS_M))

        # --- Actuators ---
        # Use 'motor' for torque-control matching the get_control_output logic
        actuator_root = ET.SubElement(root,  "actuator")
        for act_name in actuators:
            ET.SubElement(actuator_root, "motor",
                          name=f"{act_name}_motor",
                          joint=f"{act_name}_joint",
                          gear="1") # Gear ratio is handled by Python Control logic TODO: @FRM, estimate from hardware and look into what exactly does it mean that Python Control logic handles this.
        # --- Serialization
        xml_str = ET.tostring(root, encoding="utf-8")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")

    # --- Component Output for ada_assets Integration ---

    def generate_component_mjcf(self) -> str:
        """Build a standalone component XML for MjSpec.attach().

        Follow the ada_assets component pattern (cf. jaco.xml) for attachment at a named site on a parent model:
            arm_spec = mujoco.MjSpec.from_string(
                assembler.generate_component_mjcf()
            )
            spec.attach(arm_spec, prefix="", site=parent_site)

        Includes dual geometry (visual group = 2, collision group = 3),
        base_mount_site / ee_site, motor_actuators, adjacent-link, contact exclusions, and home / stow keyframes.

        Use radians and local coordinates with gravity compensation on all bodies. Omits floor, lighting, and sim options.

        Returns:
            Pretty-printed MCJCF XML string.
        """
        logger.info("generating_component_mjcf", n_modules=len(self.modules))

        root = ET.Element("mujoco", model="neuro_manipulator")
        ET.SubElement(root, "compiler", angle="radian", coordinate="local", autolimits="true")

        # --- Default classes (Menagerie pattern) ---
        defaults = ET.SubElement(root, "default")
        arm_cls = ET.SubElement(defaults, "default")
        arm_cls.set("class", "neuro_arm")
        ET.SubElement(arm_cls, "joint", damping = "1.0") # TODO: @FRM, figure out what damping is doing here

        vis_cls = ET.SubElement(arm_cls, "default")
        vis_cls.set("class", "neuro_visual")
        ET.SubElement(vis_cls, "geom", type="capsule", contype="0", conaffinity="0", group="2")

        col_cls = ET.SubElement(arm_cls, "default")
        col_cls.set("class", "neuro_collision")
        ET.SubElement(col_cls, "geom", type="capsule", contype="1", conaffinity="1", group="3", rgba="0.3 0.5 0.3 0.2")

        # --- Materials ---
        asset = ET.SubElement(root, "asset")
        for mat_name, mat_attrs in MATERIALS.items():
            ET.SubElement(asset, "material", name=mat_name, **mat_attrs)

        # --- Worldbody (ANRM chain only; no floor or lighting) ---
        worldbody = ET.SubElement(root, "worldbody")

        body_names: list[str] = []
        actuator_names: list[str] = []
        joint_ranges: list[tuple[float, float]] = []

        last_parent = worldbody

        for i, (module, link_length) in enumerate(zip(self.modules, self.link_lengths),):
            body_name = f"body_{module.name}"
            body_attribs: dict[str, str] = {
                "name": body_name,
                "pos": "0 0 0",
                "gravcomp": "1",
            }
            if i == 0:
                body_attribs["childclass"] = "neuro_arm"

            current_body = ET.SubElement(last_parent, "body", **body_attribs)
            body_names.append(body_name)
            # Attachment anchor on the root body
            if i == 0:
                ET.SubElement(
                    current_body, "site",
                    name="base_mount_site",
                    pos="0 0 0",
                    size="0.01",
                    rgba="0 0 1 0.3",
                )

            # --- Joint ---
            joint_type = "slide" if module.is_prismatic else "hinge"
            armature = (
                str(PRISMATIC_ARMATURE) if module.is_prismatic else str(JOINT_ARMATURE)
            )
            joint_kwargs: dict[str, str] = {
                "name": f"{module.name}_joint",
                "type": joint_type,
                "axis": self._array_to_str(module.axis),
                "stiffness": str(module.k_s),
                "damping": str(module.c_s),
                "armature": armature,
            }

            range_str = self._format_range(module, use_radians=True)
            if range_str is not None:
                joint_kwargs["limited"] = "true"
                joint_kwargs["range"] = range_str
                low, high = module.joint_range
                joint_ranges.append((float(low), float(high)))
            else:
                joint_ranges.append((0.0, 0.0))

            ET.SubElement(current_body, "joint", **joint_kwargs)

            # --- Inertial ---
            ET.SubElement(
                current_body, "inertial",
                pos=self._array_to_str(module.com),
                mass=str(module.m),
                diaginertia=DEFAULT_DIAGINERTIA,
            )

            # --- Link Geometry (visual + collision) ---
            geom_attrs = self._build_link_geom_attrs(module, link_length)
            if geom_attrs is not None:
                material = ("arm_base" if self._is_base_link(module) else "arm_link")

                vis_geom = ET.SubElement(current_body, "geom")
                vis_geom.set("class", "neuro_visual")
                vis_geom.set("name", f"{module.name}_link_visual")
                vis_geom.set("fromto", geom_attrs["fromto"])
                vis_geom.set("size", geom_attrs["size"])
                vis_geom.set("material", material)

                col_geom = ET.SubElement(current_body, "geom")
                col_geom.set("class", "neuro_collision")
                col_geom.set("name", f"{module.name}_link_collision")
                col_geom.set("fromto", geom_attrs["fromto"])
                col_geom.set("size", geom_attrs["size"])

            if module.is_actuated:
                actuator_names.append(module.name)

            # --- Chain to next body or terminate with EE site ---
            if i < len(self.modules) - 1:
                attach_pos = (
                    module.r_attach if np.any(module.r_attach) else np.array([link_length, 0, 0])
                )
                last_parent = ET.SubElement(current_body, "body", name=f"mount_{i}", pos=self._array_to_str(attach_pos))
            else:
                ET.SubElement(current_body, "site", name="ee_site", pos=f"{link_length} 0 0", size=str(EE_SITE_RADIUS_M), rgba="0 1 0 1")

        # --- Actuators ---
        actuator_root = ET.SubElement(root, "actuator")
        for act_name in actuator_names:
            ET.SubElement(
                actuator_root, "motor",
                name=f"{act_name}_motor",
                joint=f"{act_name}_joint",
                gear="1",
            )

        # --- Contact Exclusions (adjacent links) --- TODO: @FRM, figure out why this is important.
        contact = ET.SubElement(root, "contact")
        for j in range(len(body_names) - 1):
            ET.SubElement(
                contact, "exclude",
                body1=body_names[j],
                body2=body_names[j + 1],
            )

        # --- Keyframes ---
        self._add_keyframes(root, joint_ranges)

        logger.info(
            "generated_component_mjcf",
            n_bodies=len(body_names),
            n_actuators=len(actuator_names),
        )

        xml_str = ET.tostring(root, encoding="utf-8")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")

    def _add_keyframes(self, root: ET.Element, joint_ranges: list[tuple[float, float]]) -> None:
        """Append home and stow keyframe elements to the MJCF root.

        Args:
              root: The "<mujoco>" root element.
              joint_ranges: Per-joint (low, high) bounds in radians.
        """
        keyframe_section = ET.SubElement(root, "keyframe")

        home_vals = [(lo + hi) / 2.0 for lo, hi in joint_ranges]
        ET.SubElement(
            keyframe_section, "key",
            name="home",
            qpos=" ".join(f"{v:.4f}" for v in home_vals),
        )

        # Stow: joints near lower bounds (compact folded configuration)
        stow_vals = [lo + 0.1 * (hi - lo) for lo, hi in joint_ranges]
        ET.SubElement(
            keyframe_section, "key",
            name="stow",
            qpos=" ".join(f"{v:.4f}" for v in stow_vals),
        )

    # --- Save helpers ---

    def save_model(self, path:str = "model.xml")-> None:
        """Write full-scene MJCF to a file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            f.write(self.generate_mjcf())
        logger.info(f"scene_model_saved", path=path)

    def save_component(self, path:str = "neuro_arm.xml")-> None:
        """Write component MJCF for MjSpec.attach() to a file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            f.write(self.generate_component_mjcf())
        logger.info(f"component_model_saved", path=path)

def main() -> int:
    """Generate MCJF model files from the centralized robot config."""
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo XML for the Assistive NeuroRobot Manipulator"
    )
    parser.add_argument(
        "--mode", choices=["scene", "component", "both"],
        default="component",
        help="Output mode (default: component)",
    )
    parser.add_argument(
        "--scene-out", default="model.xml",
        help="Output path for full-scene MJCF (default: model.xml)",
    )
    parser.add_argument(
        "--component-out", default="neuro_arm.xml",
        help="Output path for component MJCF (default: neuro_arm.xml)",
    )
    args = parser.parse_args()

    from modular_arm.robot.robot_config import get_robot_config
    config = get_robot_config()
    assembler: ArmAssembler = config["assembler"]

    from modular_arm.core.config import get_paths
    mjcf_dir = get_paths().robot_mjcf_dir
    mjcf_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("scene", "both"):
        assembler.save_model(path=str(mjcf_dir / args.scene_out))
    if args.mode in ("component", "both"):
        assembler.save_component(path=str(mjcf_dir / args.component_out))
    return 0

if __name__ == "__main__":
    sys.exit(main())

