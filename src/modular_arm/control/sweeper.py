import os

import mujoco
import mujoco.viewer
import numpy as np
import time
from modular_arm.modules.sea_module import SEAModule
from modular_arm.modules.arm_assembler import ArmAssembler


def run_workspace_sweep():
    # # 1. Hardware Definition (Same as your SEAModule)
    # m1 = SEAModule(name="shoulder", mass=1.0, com=np.array([0, 0, 0.15]),
    #                axis=np.array([0, 1, 0]), k_virtual=60.0, d_virtual=4.0)
    # m2 = SEAModule(name="elbow", mass=0.8, com=np.array([0, 0, 0.15]),
    #                axis=np.array([0, 1, 0]), k_virtual=40.0, d_virtual=3.0)
    # m1.set_child(m2, attach_point=np.array([0, 0, 0.3]))

    # 1. Change axis to [0, 0, 1] (Rotation around Z-axis, like a record player)
    # This ensures the motion stays in the XY plane (parallel to ground)
    m1 = SEAModule(name="shoulder", mass=1.0, com=np.array([0.15, 0, 0]),
                   axis=np.array([0, 0, 1]), k_virtual=60.0, d_virtual=4.0)
    m2 = SEAModule(name="elbow", mass=0.8, com=np.array([0.15, 0, 0]),
                   axis=np.array([0, 0, 1]), k_virtual=40.0, d_virtual=3.0)

    # Update attachment point to be along the X-axis
    m1.set_child(m2, attach_point=np.array([0.3, 0, 0]))

    # 2. Assemble and Load
    assembler = ArmAssembler(base_position=np.array([0, 0, 0.5]))
    assembler.add_module(m1, link_length=0.3)
    assembler.add_module(m2, link_length=0.3)

    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # Storage for the envelope points
    workspace_points = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running() and (time.time() - start_time < 30):  # Sweep for 30s
            step_start = time.time()
            t = time.time() - start_time

            # --- SPACE-FILLING TRAJECTORY ---
            # Limits: Shoulder +/- 90 deg, Elbow 0 to 150 deg
            q_target_shoulder = 1.5 * np.sin(0.5 * t)
            q_target_elbow = 1.3 * (0.5 * np.sin(2.5 * t) + 0.5)

            # --- CONTROL ---
            g_vec = np.array([0, 0, -9.81])
            tau1, _, _ = m1.get_control_output(g_vec, q_target_shoulder, data.qpos[0], data.qvel[0])
            tau2, _, _ = m2.get_control_output(g_vec, q_target_elbow, data.qpos[1], data.qvel[1])

            data.ctrl[0], data.ctrl[1] = tau1, tau2

            # --- DATA CAPTURE ---
            # Record the end-effector position in world coordinates
            ee_pos = data.site_xpos[ee_site_id].copy()
            workspace_points.append(ee_pos)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time sync
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Convert to numpy for analysis
    points = np.array(workspace_points)
    print(f"Captured {len(points)} points for workspace analysis.")
    return points


if __name__ == "__main__":
    points = run_workspace_sweep()

    # 1. Define the output path
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "workspace_envelope.csv")

    # 2. Save as CSV for easy inspection/plotting
    np.savetxt(output_path, points, delimiter=",", header="x,y,z", comments="")

    print(f"Workspace log saved to: {output_path}")