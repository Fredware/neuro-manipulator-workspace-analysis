import mujoco
import numpy as np
import polars as pl
import os
import time

from lxml.html.defs import head_tags
from tqdm import tqdm
from modular_arm.core.robot_config import get_robot_config

def run_stratified_montecarlo(samples_per_layer=5000):
    # 1. Load robot params
    config = get_robot_config()
    assembler = config["assembler"]
    modules = config["modules"]

    # Generate the physics model
    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # --- Define Stratification Layers (0%, 25%, 50%, 75%, 100% height) ---
    max_slide = config["slide_range_meters"]
    height_layers = np.linspace(0, max_slide, 5)
    layer_labels = ["0%", "25%", "50%", "75%", "100%"]

    results = []

    total_samples = samples_per_layer * len(height_layers)
    print(f"Executing Stratified Monte Carlo")
    print(f"Layers: 5 | Samples per layer: {samples_per_layer} | Total samples: {total_samples}")

    start_time = time.time()

    for layer_idx, h_val in enumerate(height_layers):
        label = layer_labels[layer_idx]

        for _ in tqdm(range(samples_per_layer), desc=f"Layer {label} ({h_val:.3f}m)"):
            q_rand = []

            # --- Dynamically pull joint limits from each module ---
            for i, mod in enumerate(modules):
                if mod.name == "dof2_slide_z":
                    # explicitly lock the passive prismatic joint to the current strata height
                    q_rand.append(h_val)
                else:
                    # randomly sample active rotary/bend joint states within bounds
                    low, high = mod.joint_range
                    q_rand.append(np.random.uniform(low, high))
            # Inject random joint position directly into the physics state
            data.qpos[:6] = q_rand

            # Run forward kinematics (compute global positions without simulating time/dynamics)
            mujoco.mj_kinematics(model, data)

            # capture end-effector position
            ee_pos = data.site_xpos[ee_site_id].copy()

            # Log all spatial and kinematic data
            results.append({
                "x": ee_pos[0], "y": ee_pos[1], "z": ee_pos[2],
                "layer_label": label, "layer_height": h_val,
                "q1": q_rand[0], "q2": q_rand[1], "q3": q_rand[2],
                "q4": q_rand[3], "q5": q_rand[4], "q6": q_rand[5],
            })
    # --- Serialize to polars dataframe ---
    elapsed_time = time.time() - start_time
    df = pl.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    save_path = "data/monte_carlo_results_6dof.csv"
    df.write_csv(save_path)

    print(f"\n Stratified 6DOF Workspace Dataset Generated")
    print(f"Speed:  {total_samples / elapsed_time:.2f} samples/sec")
    print(f"Saved to: {save_path}")

def run_monte_carlo(samples=5000):
    config = get_robot_config()
    assembler = config["assembler"]
    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # 2. Define Parametric Ranges (Rad)
    ranges = [np.deg2rad([0, 160]), np.deg2rad([0, 160])]

    results = []

    print(f"Sampling {samples} points in Configuration Space...")
    for _ in tqdm(range(samples)):
        # Random Sample in Joint Space
        q_rand = [np.random.uniform(r[0], r[1]) for r in ranges]

        # Set MuJoCo state and compute Forward Kinematics
        data.qpos[:2] = q_rand
        mujoco.mj_forward(model, data)

        # Capture EE position
        ee_pos = data.site_xpos[ee_site_id].copy()
        results.append({
            "x": ee_pos[0], "y": ee_pos[1], "z": ee_pos[2],
            "q1": q_rand[0], "q2": q_rand[1]
        })

    # 3. Save to data/
    df = pl.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    df.write_csv("data/monte_carlo_results.csv")
    print(f"✅ Monte Carlo analysis saved to data/monte_carlo_results.csv")


if __name__ == "__main__":
    # run_monte_carlo()
    run_stratified_montecarlo()
