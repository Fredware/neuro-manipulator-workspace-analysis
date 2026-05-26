import numpy as np
from modular_arm.modules.sea_module import SEAModule
from modular_arm.modules.arm_assembler import ArmAssembler

def get_robot_config():
    """
    Centralized robot definition.
    Changes in robot parameters in this module propagate to the designer and analysis scripts.
    """
    # --- Physical SEA Module Definition ---
    m1 = SEAModule(
        name = "proximal",
        mass = 1.2, # [kg]
        com  = np.array([0.15, 0, 0]),
        axis = np.array([0, 0, 1]), # Horizontal rotation
        k_virtual = 60.0,
        d_virtual = 5.0,
        k_spring  = 150.0 # Physical SEA stiffness
    )

    m2 = SEAModule(
        name = "distal",
        mass = 0.7,
        com = np.array([0.1, 0, 0]),
        axis = np.array([0, 0, 1]),
        k_virtual = 40.0,
        d_virtual = 4.0,
        k_spring = 100
    )

    # --- Tree Structure Definition ---
    m1.set_child(m2, attach_point=np.array([0.35, 0, 0]))

    # --- Setup Assembler ---
    assembler = ArmAssembler(base_position=np.array([0, 0, 0.5]))
    assembler.add_module(m1, link_length=0.35)
    assembler.add_module(m2, link_length=0.30)

    return {
        "modules": [m1, m2],
        "assembler": assembler,
        "joint_ranges_deg": {
            "proximal": [0, 160],
            "distal": [0, 160]
        }
    }

