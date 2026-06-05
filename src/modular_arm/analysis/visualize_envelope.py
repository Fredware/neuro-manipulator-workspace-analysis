import polars as pl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

from modular_arm.core.robot_config import get_robot_config


# --- HELPER FUNCTIONS ---
def draw_sphere(ax, center, radius, color, alpha=0.25):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor=None)


def draw_robot_base(ax, origin, axis_length=0.2):
    """Draws the robot origin and RGB coordinate axes."""
    ox, oy, oz = origin

    # Draw Origin Point
    ax.scatter([ox], [oy], [oz], color='black', s=50, marker='o', label="Robot Base")

    # X-Axis (Red)
    ax.quiver(ox, oy, oz, axis_length, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox + axis_length * 1.1, oy, oz, 'X+', color='red', fontweight='bold')

    # Y-Axis (Green)
    ax.quiver(ox, oy, oz, 0, axis_length, 0, color='green', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox, oy + axis_length * 1.1, oz, 'Y+', color='green', fontweight='bold')

    # Z-Axis (Blue)
    ax.quiver(ox, oy, oz, 0, 0, axis_length, color='blue', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox, oy, oz + axis_length * 1.1, 'Z+', color='blue', fontweight='bold')


def draw_manipulability_field(ax, df, model, data, ee_site_id, num_samples=15):
    """
    Samples configurations and draws the principal axes of motion (Manipulability Field).
    """
    print(f"Calculating Jacobian field for {num_samples} samples...")

    # 1. Randomly sample N configurations from the Monte Carlo results
    sampled_df = df.sample(n=num_samples)

    # Pre-allocate Jacobian array (3 translation DOFs x N joints)
    jacp = np.zeros((3, model.nv))

    for row in sampled_df.iter_rows(named=True):
        # 2. Set the MuJoCo state to this specific configuration
        data.qpos[0] = row["q1"]
        data.qpos[1] = row["q2"]
        mujoco.mj_kinematics(model, data)  # Update global positions
        mujoco.mj_comPos(model, data)  # Update Center of Mass

        # 3. Calculate the Translation Jacobian for the End-Effector
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)

        # 4. Perform SVD to get the principal directions of motion
        U, S, Vh = np.linalg.svd(jacp)

        # Base position of the arrow
        x, y, z = row["x"], row["y"], row["z"]

        # 5. Draw the principal axes (using the first 2 singular values for a 2-DOF arm)
        # S[0] is the primary direction of motion, S[1] is the secondary
        for i in range(len(S)):
            if S[i] < 1e-4:  # Ignore singular directions (dead zones)
                continue

            # U[:, i] is the 3D direction vector. S[i] is the magnitude.
            direction = U[:, i] * S[i] * 0.5  # Scale by 0.5 for visual appeal

            dx, dy, dz = direction[0], direction[1], direction[2]

            # Color code: Primary axis (easiest movement) = Blue, Secondary = Orange
            color = 'blue' if i == 0 else 'orange'
            linewidth = 3 if i == 0 else 1.5

            # Plot the arrow (both positive and negative to show bidirectional capability)
            ax.quiver(x, y, z, dx, dy, dz, color=color, linewidth=linewidth, arrow_length_ratio=0.2)
            ax.quiver(x, y, z, -dx, -dy, -dz, color=color, linewidth=linewidth, arrow_length_ratio=0.0)

            # Plot a small black dot at the center for clarity
            ax.scatter(x, y, z, color='black', s=10)

def draw_orientation_field(ax, df, model, data, ee_site_id, num_samples=25):
    """
    Samples configurations and draws a fixed-length arrow representing
    the hand/tool orientation (Orientation Workspace).
    """
    print(f"Calculating Orientation field for {num_samples} samples...")

    # 1. Randomly sample N configurations
    sampled_df = df.sample(n=num_samples)

    # Define a fixed, small length for the arrows (e.g., 5 cm)
    ARROW_LENGTH = 0.05

    for row in sampled_df.iter_rows(named=True):
        # 2. Set the MuJoCo state to this specific configuration
        data.qpos[0] = row["q1"]
        data.qpos[1] = row["q2"]
        mujoco.mj_kinematics(model, data)

        # 3. Get the Rotation Matrix of the End-Effector Site
        # MuJoCo exposes this as a flattened 9-element array, so we reshape it
        rot_mat = data.site_xmat[ee_site_id].reshape(3, 3)

        # 4. Extract the pointing vector.
        # By default in MuJoCo, sites point along their local Z-axis.
        # If your "flashlight" points along X, use rot_mat[:, 0] instead.
        pointing_vector = rot_mat[:, 2]

        dx, dy, dz = pointing_vector[0], pointing_vector[1], pointing_vector[2]
        x, y, z = row["x"], row["y"], row["z"]

        # 5. Draw a single, clean arrow pivoting from the tail
        ax.quiver(x, y, z, dx, dy, dz,
                  length=ARROW_LENGTH,
                  normalize=True,  # Ensures it stays exactly ARROW_LENGTH
                  pivot='tail',  # Arrow starts exactly at x,y,z
                  color='purple',
                  linewidth=2,
                  arrow_length_ratio=0.3)

        # Draw the physical (x,y,z) coordinate as a dot
        ax.scatter(x, y, z, color='black', s=15, zorder=5)

def visualize_workspace(file_name="monte_carlo_results.csv", robot_base=[0, 0, 0.5]):
    file_path = os.path.join("data", file_name)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    df = pl.read_csv(file_path)

    # --- Define ADL Zone A (Personal Care) ---
    zone_a = {'x': [0.1, 0.4], 'y': [-0.2, 0.2], 'z': [0.4, 0.6]}

    # --- Setup Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- Draw Robot Base Frame ---
    draw_robot_base(ax, robot_base, axis_length=0.15)

    # --- Draw Zone A box ---
    xb, yb, zb = zone_a['x'], zone_a['y'], zone_a['z']
    for i in range(2):
        for j in range(2):
            ax.plot([xb[i], xb[i]], [yb[j], yb[j]], zb, color='red', alpha=0.5, linewidth=2)
            ax.plot([xb[i], xb[i]], yb, [zb[i], zb[i]], color='red', alpha=0.5, linewidth=2)
            ax.plot(xb, [yb[i], yb[i]], [zb[i], zb[i]], color='red', alpha=0.5, linewidth=2)

    # --- Identify Hits for Capability Map ---
    hits = df.with_columns(
        in_zone=(
                pl.col("x").is_between(xb[0], xb[1]) &
                pl.col("y").is_between(yb[0], yb[1]) &
                pl.col("z").is_between(zb[0], zb[1])
        )
    )

    is_monte_carlo = len(df) > 500

    if is_monte_carlo:
        # --- MODE 1: Point Cloud (High-Res) ---
        print(f"Rendering Point Cloud ({len(df)} points)...")
        in_zone_df = hits.filter(pl.col("in_zone"))
        out_zone_df = hits.filter(~pl.col("in_zone"))

        ax.scatter(out_zone_df["x"], out_zone_df["y"], out_zone_df["z"], c='darkgray', s=2, alpha=0.3)
        ax.scatter(in_zone_df["x"], in_zone_df["y"], in_zone_df["z"], c='limegreen', s=10, alpha=0.8)

        coverage = (len(in_zone_df) / len(df)) * 100
        print(f"--- ADL INTERSECTION SCORE ---")
        print(f"Total Configurations: {len(df)}")
        print(f"Zone A Hits: {len(in_zone_df)}")
        print(f"Coverage Metric: {coverage:.2f}%")
        ax.set_title(f"Monte Carlo Reachability (Zone A Coverage: {coverage:.2f}%)")

    else:
        # --- MODE 2: Voxels (Low-Res) ---
        print(f"Rendering Voxels ({len(df)} points)...")
        VOXEL_RADIUS = 0.05
        for row in hits.iter_rows(named=True):
            point = [row["x"], row["y"], row["z"]]
            color = 'limegreen' if row["in_zone"] else 'darkgray'
            draw_sphere(ax, point, VOXEL_RADIUS, color, alpha=0.4)
        ax.set_title("Discrete Workspace Voxelization")

    config = get_robot_config()
    assembler = config["assembler"]
    model = mujoco.MjModel.from_xml_string(assembler.generate_mjcf())
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    # draw_manipulability_field(ax, hits, model, data, ee_site_id, num_samples=15)
    draw_orientation_field(ax, hits, model, data, ee_site_id, num_samples=25)

    # --- Plot formatting ---
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_box_aspect([1, 1, 1])

    # Dynamic scaling including the base position
    pts = df.select(["x", "y", "z"]).to_numpy()
    all_x = np.append(pts[:, 0], robot_base[0])
    all_y = np.append(pts[:, 1], robot_base[1])
    all_z = np.append(pts[:, 2], robot_base[2])

    buffer = 0.2
    ax.set_xlim(all_x.min() - buffer, all_x.max() + buffer)
    ax.set_ylim(all_y.min() - buffer, all_y.max() + buffer)

    z_min, z_max = all_z.min(), all_z.max()
    if z_max - z_min < 0.1:
        ax.set_zlim(z_min - 0.2, z_max + 0.2)
    else:
        ax.set_zlim(z_min - buffer, z_max + buffer)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    visualize_workspace("monte_carlo_results.csv")