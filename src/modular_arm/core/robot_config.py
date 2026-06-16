import numpy as np
from modular_arm.modules.sea_module import SEAModule
from modular_arm.modules.arm_assembler import ArmAssembler

IN_TO_M = 0.0254

def get_robot_config():
    """
    Centralized robot definition.
    Changes in robot parameters in this module propagate to the designer and analysis scripts.
    """
    # --- Centralized Customizable Lengths (in inches) ---
    LENGHTS = {
        "dof1_height": 11.5,
        "dof2_slide_range": 7.0,
        "dof2_start_height": 2.6,
        "dof3_length": 8.5,
        "dof4_length": 6.0,
        "dof6_length": 3.0,
        "ee_length": 1.5,
    }
    # convert to SI units for MuJoCo
    m_lengths = {k: v * IN_TO_M for k, v in LENGHTS.items()}

    # --- Physical SEA Module Definition ---
    # DOF 1: Base Rotary (Axis Z, Points UP)
    m1 = SEAModule("dof1_rot_z", mass=1.5, com=np.array([0, 0, m_lengths["dof1_height"]/2]), axis=np.array([0, 0, 1]), k_spring=150, k_virtual=200.0, d_virtual=20.0)
    m1.link_direction = np.array([0, 0, 1]) # Extend up
    m1.joint_range = np.deg2rad([0, 360])

    # DOF 2: Passive Prismatic (Axis Z, Sliding Inside Cylinder)
    m2 = SEAModule("dof2_slide_z", mass = 0.8, com=np.array([0, 0, 0]), axis=np.array([0, 0, 1]), is_prismatic=True, is_actuated=False)
    m2.link_direction = np.array([0, 0, 0]) # Internal joint, no extra structural mesh length
    m2.joint_range = np.array([0, m_lengths["dof2_slide_range"]])
    m2.r_attach = np.array([0, 0, m_lengths["dof2_start_height"]])

   # DOF 3: First Forward Bend (Axis Z, Link points along X)
    m3 = SEAModule("dof3_bend_z", mass=1.1, com=np.array([m_lengths["dof3_length"]/2, 0, 0]), axis=np.array([0, 0, 1]), k_spring=150, k_virtual=200.0, d_virtual=20.0)
    m3.link_direction = np.array([1, 0, 0]) # Extend forward
    m3.joint_range = np.deg2rad([30, 150])

    # DOF 4: Second Forward Bend (Axis Z, Link points along X)
    m4 = SEAModule("dof4_bend_z", mass=0.9, com=np.array([m_lengths["dof4_length"]/2, 0, 0]), axis=np.array([0, 0, 1]), k_spring=150, k_virtual=100.0, d_virtual=10.0)
    m4.link_direction = np.array([1, 0, 0])
    m4.joint_range = np.deg2rad([30, 150])

    # DOF 5: Wrist Roll (Axis X, Twist Joint)
    m5 = SEAModule("dof5_roll_x", mass=0.5, com=np.array([0, 0, 0]), axis=np.array([1, 0, 0]), k_spring=150, k_virtual=60.0, d_virtual=5.0)
    m5.link_direction = np.array([1, 0, 0])
    m5.joint_range = np.deg2rad([30, 150])

    # DOF 6: Wrist Pitch/Yaw Bend (Axis Z, Link points along X)
    m6 = SEAModule("dof6_bend_z", mass=0.4, com=np.array([m_lengths["dof6_length"]/2, 0, 0]), axis=np.array([0, 0, 1]), k_spring=150, k_virtual=60.0, d_virtual=5.0)
    m6.link_direction = np.array([1, 0, 0])
    m6.joint_range = np.deg2rad([30, 150])

    # --- Tree Structure Definition / Kinematic Chain Linking ---
    m1.set_child(m2, m2.r_attach)
    m2.set_child(m3, np.array([0.05, 0, 0]))  # Slight radial offset to clear the base cylinder wall
    m3.set_child(m4, np.array([m_lengths["dof3_length"], 0, 0]))
    m4.set_child(m5, np.array([m_lengths["dof4_length"], 0, 0]))
    m5.set_child(m6, np.array([0.02, 0, 0]))  # Small structural housing offset

    # --- Procedural Assembler Construction ---
    assembler = ArmAssembler(base_position=np.array([0, 0, 0.0]))
    assembler.add_module(m1, m_lengths["dof1_height"])
    assembler.add_module(m2, 0)  # Prismatic segment
    assembler.add_module(m3, m_lengths["dof3_length"])
    assembler.add_module(m4, m_lengths["dof4_length"])
    assembler.add_module(m5, 0)  # Pure twist joint
    assembler.add_module(m6, m_lengths["dof6_length"] + m_lengths["ee_length"])

    return {
        "modules": [m1, m2, m3, m4, m5, m6],
        "assembler": assembler,
        "slide_range_meters": m_lengths["dof2_slide_range"],
    }

