import os

import mujoco
import mujoco.viewer
import numpy as np
import time
import polars as pl

from dm_control.utils.rewards_test import ToleranceTest
from modular_arm.modules.sea_module import SEAModule
from modular_arm.modules.arm_assembler import ArmAssembler
from modular_arm.core.robot_config import get_robot_config

TOLERANCE = 0.05 # Radians to consider a point "reached". TODO: move to config file

def run_parametric_tree_sweep():
    config = get_robot_config()
    assembler = config['assembler']
    m1, m2 = config['modules']

    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # --- Generate Grid Search Targets ---
    m1_pts = np.deg2rad(np.linspace(*config['joint_ranges_deg']['proximal'], 12)) # 12 partitions TODO: move to config file
    m2_pts = np.deg2rad(np.linspace(*config['joint_ranges_deg']['distal'], 12))

    target_list = []
    for p1 in m1_pts:
        for p2 in m2_pts:
            target_list.append([p1, p2])

    current_target_idx = 0
    workspace_points = [] # Storage for the envelope points
    target_start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running() and (current_target_idx < len(target_list)):  # Sweep for 30s
            step_start = time.time()
            t = time.time() - start_time

            q_target = target_list[current_target_idx]

            # --- CONTROL ---
            g_vec = np.array([0, 0, -9.81])
            tau1, _, _ = m1.get_control_output(g_vec, q_target[0], data.qpos[0], data.qvel[0])
            tau2, _, _ = m2.get_control_output(g_vec, q_target[1], data.qpos[1], data.qvel[1])
            data.ctrl[0], data.ctrl[1] = tau1, tau2

            # --- STATE TRANSITION LOGIC ---
            error = np.linalg.norm(data.qpos[:2] - q_target)
            time_on_target = time.time()- target_start_time
            # Transition reached or if it timed out (e.g., stuck for 5 seconds)
            if error < TOLERANCE or time_on_target > 5.0:
                # Capture EE position at this grid point
                ee_pos = data.site_xpos[ee_site_id].copy()

                # Tag as reached or failed
                status = "REACHED" if error < TOLERANCE else "TIMEOUT"
                workspace_points.append({"x": ee_pos[0], "y": ee_pos[1], "z": ee_pos[2]})

                print(f"Target {current_target_idx+1}/{len(target_list)} | {status} | Control Error: {error:.3f}")
                current_target_idx += 1
                target_start_time = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time sync
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # --- Save with Polars
    if workspace_points:
        df = pl.DataFrame(workspace_points)
        os.makedirs("data", exist_ok=True)
        df.write_csv("data/workspace_envelope.csv")
        print("Grid Search Analysis COMPLETE")

if __name__ == "__main__":
    run_parametric_tree_sweep()

