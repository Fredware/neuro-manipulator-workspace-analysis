import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import polars as pl

from modular_arm.robot.robot_config import get_robot_config

TOLERANCE = 0.05 # Radians to consider a point "reached". TODO: move to config file

def run_parametric_tree_sweep():
    # --- Load robot config ---
    config = get_robot_config()
    assembler = config['assembler']
    modules = config['modules'] # n = 6 spatial modules

    # --- Compile XML string and load MuJoCo entities ---
    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # --- Set initial state of the passive slider to the top ---
    slider_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dof2_slide_z_joint")
    slider_q_idx = model.jnt_qposadr[slider_jnt_id]
    data.qpos[slider_q_idx] = config["slide_range_meters"]
    mujoco.mj_forward(model, data) # Update the physics to reflect the new position

    # --- Generate Grid Search Targets Dynamically from Module Properties ---
    grid_points = []
    for mod in modules:
        # Use 3 partitions (min, median, max boundary)
        # For the passive joint, use structural slide range limits
        pts = np.linspace(mod.joint_range[0], mod.joint_range[1], 2)
        grid_points.append(pts)
    # Use meshgrid to expand all n-dimensional combinations (6DOF Tree Expansion)
    mesh = np.meshgrid(*grid_points, indexing='ij')
    target_list = np.vstack([m.flatten() for m in mesh]).T

    current_target_idx = 0
    workspace_points = [] # Storage for the envelope points
    target_start_time = time.time()

    # --- Physics Simulation ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running() and (current_target_idx < len(target_list)):  # Sweep for 30s
            step_start = time.time()
            q_target = target_list[current_target_idx]

            # --- DYNAMIC IMPEDANCE CONTROL LOOP ---
            g_vec = np.array([0, 0, -9.81])

            for i, mod in enumerate(modules):
                # Safely map tracking parameters via MuJoCo's structural register ID
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{mod.name}_joint")
                q_idx = model.jnt_qposadr[jnt_id]
                v_idx = model.jnt_dofadr[jnt_id]

                # Fetch real-time hardware states
                q_curr = data.qpos[q_idx]
                v_curr = data.qvel[v_idx]

                # Compute individual motor control laws
                tau, _, _ = mod.get_control_output(g_vec, q_target[i], q_curr, v_curr)

                # Assign torque only if a matching hardware motor exists (skip passive slider)
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{mod.name}_motor")
                if act_id != -1:
                    data.ctrl[act_id] = float(tau)
                elif mod.name == "dof2_slide_z":
                    # --- Hacky simulation of the "set-screw" ---
                    # If it's the passive slider, physically lock it at the target height so gravity cannot pull it down
                    data.qpos[q_idx] = q_target[i]
                    data.qvel[v_idx] = 0.0 # Kill any falling velocity

            # --- STATE TRANSITION LOGIC ---
            error = np.linalg.norm(data.qpos[:6] - q_target)
            time_on_target = time.time() - target_start_time

            # Transition reached or if it timed out (e.g., stuck for 5 seconds)
            if error < TOLERANCE or time_on_target > 3.0:
                # Capture EE position at this grid point
                ee_pos = data.site_xpos[ee_site_id].copy()
                # Tag as reached or failed
                status = "REACHED" if error < TOLERANCE else "TIMEOUT"
                # Append full 3D spatial values for downstream mapping tools
                workspace_points.append({
                    "x": ee_pos[0],
                    "y": ee_pos[1],
                    "z": ee_pos[2],
                    "status": status
                })

                print(f"Target {current_target_idx+1}/{len(target_list)} | {status} | Control Error: {error:.3f}")
                current_target_idx += 1
                target_start_time = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time sync
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # --- Save / Serialize Target Coordinate Outputs with Polars
    if workspace_points:
        df = pl.DataFrame(workspace_points)
        os.makedirs("data", exist_ok=True)
        df.write_csv("data/workspace_envelope.csv")
        print("Grid Search Analysis COMPLETE")

if __name__ == "__main__":
    run_parametric_tree_sweep()

