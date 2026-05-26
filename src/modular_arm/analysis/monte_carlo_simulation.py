import mujoco
import numpy as np
import polars as pl
import os

from fontTools.ttLib.tables.S__i_l_f import assemble
from tqdm import tqdm  # Great for tracking high-res samples
from modular_arm.modules.sea_module import SEAModule
from modular_arm.modules.arm_assembler import ArmAssembler
from modular_arm.core.robot_config import get_robot_config


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
    run_monte_carlo()

