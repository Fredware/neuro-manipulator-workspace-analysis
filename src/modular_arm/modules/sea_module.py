from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import structlog

logger = structlog.get_logger(__name__)

class SEAModule:
    """
    Represents a modular Series Elastic Actuator (SEA) joint.

    handles the recursive dynamics for gravity compensation and implements the impedance control law for modular robotic
    chains.
    """

    def __init__(self,
                 name: str,
                 mass: float,
                 com: npt.NDArray[np.float64],
                 axis: npt.NDArray[np.float64],
                 k_spring: float = 100.0, # Physical spring constant
                 c_spring: float = 1.0, # Physical damping
                 k_virtual: float = 10.0,
                 d_virtual: float = 1.0,
                 gear_ratio: float = 50.0, # Reduction ratio
                 is_actuated: bool = True,  # False for passive/fixed joints
                 is_prismatic: bool = False):
        """
        Initialize the SEA module. with physical and virtual params
        :param name: Unique identifier of the module
        :param mass: Mass of the module in [kg]
        :param com: 3D vector (x, y, z) from joint to Center of Mass
        :param axis: 3D unit vector defining the axis of motion
        :param k_spring: Physical hardware stiffness
        :param c_spring: Physical hardware damping
        :param k_virtual: Software-defined virtual stiffness for impedance control
        :param d_virtual: Software-defined virtual damping for impedance control
        :param gear_ratio: Mechanical reduction ratio
        :param is_actuated: Whether the joint has an active motor
        :param is_prismatic: True for sliding joints, False for revolute joints
        """
        self.name = name
        self.m = mass
        self.com = com
        self.axis = axis / np.linalg.norm(axis)

        # SEA Physical Properties
        self.k_s = k_spring
        self.c_s = c_spring
        self.N = gear_ratio

        self.k_virtual = k_virtual
        self.d_virtual = d_virtual

        self.is_actuated = is_actuated
        self.is_prismatic = is_prismatic
        self.joint_range = None

        self.child: Optional["SEAModule"] = None
        self.r_attach: npt.NDArray[np.float64] = np.zeros(3)
        self.link_direction = np.array([1, 0, 0])

    def set_child(self, child_module: "SEAModule", attach_point: npt.NDArray[np.float64]) -> None:
        """
        Connects a downstream module to this one.
        :param child_module:
        :param attach_point:
        :return:
        """
        self.child = child_module
        self.r_attach = attach_point

    def get_gravity_dynamics(self, g_vector: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes recursive forces and torques for gravity compensation.
        :param g_vector:
        :return:
        """
        f_downstream = np.zeros(3)
        tau_downstream = np.zeros(3)

        if self.child is not None:
            f_downstream, tau_downstream = self.child.get_gravity_dynamics(g_vector)

        weight_vector = self.m * g_vector
        f_total = weight_vector + f_downstream

        tau_local = np.cross(self.com, weight_vector)
        if self.child is not None:
            tau_total = tau_local + np.cross(self.r_attach, f_downstream) + tau_downstream
        else:
            tau_total = tau_local

        return f_total, tau_total

    def get_control_output(
            self,
            g_vector: npt.NDArray[np.float64],
            q_target: float,
            q_current: float,
            q_vel: float) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calculates the scalar command (torque or force) for this module
        :param g_vector:
        :param q_target:
        :param q_current:
        :param q_vel:
        :return:
        """
        f_total, tau_total = self.get_gravity_dynamics(g_vector)

        g_comp = (tau_total @ self.axis) if not self.is_prismatic else (f_total @ self.axis)

        if not self.is_actuated:
            return 0.0, f_total, tau_total

        # PD Impedance Control Law
        tau_impedance = (self.k_virtual * (q_target - q_current)) - (self.d_virtual * q_vel)

        # New: Spring Feed-forward compensation
        # We know the physical spring will resist by (k_s * q_current)
        # We cancel it out so the impedance controller only deals with the task.
        tau_feedforward = self.k_s * q_current
        tau_cmd = tau_impedance + g_comp + tau_feedforward

        return tau_cmd, f_total, tau_total

