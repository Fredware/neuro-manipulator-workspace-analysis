from typing import List
from modular_arm.modules.sea_module import SEAModule
from xml.dom import minidom
import numpy as np
import numpy.typing as npt
import xml.etree.ElementTree as ET


class ArmAssembler:
    """
    Procedurally generate MuJoCo models from a list of SEA modules.
    """
    def __init__(self, base_position: npt.NDArray[np.floating] ) -> None:
        self.base_position = base_position
        self.modules: List[SEAModule] = []
        self.link_lengths: List[float] = []

    def add_module(self, module: SEAModule, link_length: float) -> None:
        """
        Adds a module to the chain with a specific link length.
        :param module:
        :param link_length:
        :return:
        """
        self.modules.append(module)
        self.link_lengths.append(link_length)

    def _array_to_str(self, arr: npt.NDArray) -> str:
        """
        Helper function to convert numpy array to string.
        :param arr:
        :return:
        """
        return " ".join(map(str, arr))

    def generate_mjcf(self) -> str:
        """
        Build a MuJoCo XML string.

        Iterate through self.modules, creating <body>  tags, <joint> tags for the SEA, and <geom> tags for the links.
        :return:
        """
        # --- Create root and defaults ---
        root = ET.Element("mujoco", model="neuro_manipulator")

        # --- Global settings ---
        ET.SubElement(root, "compiler", angle="degree", coordinate="local", autolimits="true")
        option = ET.SubElement(root, "option", integrator="RK4", timestep="0.002")
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
        actuators = []

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
            # Apply range if defined
            if hasattr(module, "joint_range") and module.joint_range is not None:
                low, high = np.rad2deg(module.joint_range_rad) # MuJoCo XML expects degrees if compiler angle="degree"
                joint_kwargs["range"] = f"{low}-{high}"

            ET.SubElement(current_body, "joint", **joint_kwargs)

            # 3. define Inertial Physics
            # add small diaginertia to prevent mjMINVAL error
            # in the real model, this would be calculated from geometry
            ET.SubElement(current_body, "inertial",
                          pos=self._array_to_str(module.com),
                          mass=str(module.m),
                          diaginertia="0.01 0.01 0.01") # TODO:@FRM figure out how to estimate inertias

            # 4. define Link Geometry
            # Dynamically project capsule geometry based on custom link directions
            dir_vec = getattr(module, "link_direction", np.array([1, 0, 0]))
            end_point = dir_vec * link_length
            fromto_str = f"0 0 0 {end_point[0]} {end_point[1]} {end_point[2]}"
            ET.SubElement(current_body, "geom",
                          name=f"{module.name}_link ",
                          type="capsule",
                          fromto=fromto_str,
                          size="0.025" if "dof1" not in module.name else "0.033",
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
                ET.SubElement(current_body, "site", name="ee_site", pos=f"{link_length} 0 0", size="0.01")

        # --- Actuators ---
        # Use 'motor' for torque-control matching the get_control_output  logic
        actuator_root = ET.SubElement(root,  "actuator")
        for act_name in actuators:
            ET.SubElement(actuator_root, "motor",
                          name=f"{act_name}_motor",
                          joint=f"{act_name}_joint",
                          gear="1") # Gear ratio is handled by Python Control logic
        # --- Serialization
        xml_str = ET.tostring(root, encoding="utf-8")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")

    def save_model(self, path:str = "model.xml")-> None:
        """
        Convenience method to dump the MJCF to a file
        :param self:
        :param path:
        :return:
        """
        with open(path, "w") as f:
            f.write(self.generate_mjcf())
        print(f"Successfully saved MJCF to: {path}")

if __name__ == "__main__":
    from modular_arm.modules.sea_module import SEAModule
    import numpy as np

    # Define a simple 2-link arm
    m1 = SEAModule(name="shoulder", mass=1.0, com=np.array([0.0, 0.0, 0.1]), axis=np.array([0, 1, 0]))
    m2 = SEAModule(name="elbow", mass=0.8, com=np.array([0.0, 0.0, 0.1]), axis=np.array([0, 1, 0]))

    assembler = ArmAssembler(base_position=np.array([0, 0, 0]))
    assembler.add_module(m1, link_length=0.30)
    assembler.add_module(m2, link_length=0.30)

    assembler.save_model("test_arm.xml")