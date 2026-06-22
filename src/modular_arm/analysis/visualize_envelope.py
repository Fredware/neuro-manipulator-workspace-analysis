import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import mujoco
from modular_arm.core.robot_config import get_robot_config


# --- HELPER FUNCTIONS ---
def draw_sphere(ax, center, radius, color, alpha=0.25):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')


def draw_robot_base(ax, origin, axis_length=0.2):
    """Draws the robot origin and RGB coordinate axes."""
    ox, oy, oz = origin

    # Draw Origin Point
    ax.scatter([ox], [oy], [oz], color='black', s=50, marker='o', label="Robot Base")

    # X-Axis (Red) - Forward
    ax.quiver(ox, oy, oz, axis_length, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox + axis_length * 1.1, oy, oz, 'X+', color='red', fontweight='bold')

    # Y-Axis (Green) - Left
    ax.quiver(ox, oy, oz, 0, axis_length, 0, color='green', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox, oy + axis_length * 1.1, oz, 'Y+', color='green', fontweight='bold')

    # Z-Axis (Blue) - Up
    ax.quiver(ox, oy, oz, 0, 0, axis_length, color='blue', arrow_length_ratio=0.15, linewidth=2)
    ax.text(ox, oy, oz + axis_length * 1.1, 'Z+', color='blue', fontweight='bold')


def draw_robot_wireframe(ax, model, data, qpos=None):
    """Draws a wireframe of the robot in a specific configuration."""
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)

    # Draw links by connecting parent and child body positions
    for i in range(1, model.nbody):
        parent_id = model.body_parentid[i]
        p1 = data.xpos[parent_id]
        p2 = data.xpos[i]

        # Draw Link
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=4, alpha=0.8)
        # Draw Joint/Body Node
        ax.scatter(p2[0], p2[1], p2[2], color='darkorange', s=40, zorder=5, edgecolor='black')

    # Draw end-effector site if it exists
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_id != -1:
        ee_pos = data.site_xpos[ee_id]
        parent_body = model.site_bodyid[ee_id]
        p1 = data.xpos[parent_body]
        ax.plot([p1[0], ee_pos[0]], [p1[1], ee_pos[1]], [p1[2], ee_pos[2]], color='gray', linewidth=3, linestyle='--')
        ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], color='red', marker='X', s=100, zorder=6, label="End Effector")


def visualize_workspace(file_name="monte_carlo_results_6dof.csv", robot_base=[0, 0, 0]):
    file_path = os.path.join("data", file_name)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run the Monte Carlo simulation first!")
        return
    df = pl.read_csv(file_path)

    # --- REFERENCE FRAME TRANSLATION ---
    # Where is the human's sternum (0,0,0 for ADLs) relative to the robot base?
    # Example: Human sits 15cm behind (X), 30cm to the left (Y), and 40cm higher (Z) than the robot base.
    human_origin = np.array([-0.15, 0.30, 0.40])

    # --- Define ADL Zone A (Personal Care) in Human Frame ---
    # These are the raw empirical bounds from your .mat file analysis relative to the sternum
    zone_a_human_frame = {'x': [0.0, 0.25], 'y': [-0.2, 0.2], 'z': [0.0, 0.35]}

    # Translate the Zone into the Robot's Coordinate Frame
    zone_a = {
        'x': [zone_a_human_frame['x'][0] + human_origin[0], zone_a_human_frame['x'][1] + human_origin[0]],
        'y': [zone_a_human_frame['y'][0] + human_origin[1], zone_a_human_frame['y'][1] + human_origin[1]],
        'z': [zone_a_human_frame['z'][0] + human_origin[2], zone_a_human_frame['z'][1] + human_origin[2]]
    }

    # --- Setup Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- Draw Robot Base Frame ---
    draw_robot_base(ax, robot_base, axis_length=0.15)

    # --- Draw Human Origin (Sternum) ---
    ax.scatter(*human_origin, color='magenta', s=70, marker='*', label="Human Sternum (ADL Origin)")

    # --- Draw Robot Wireframe (Home Pose) ---
    try:
        config = get_robot_config()
        model = mujoco.MjModel.from_xml_string(config["assembler"].generate_mjcf())
        data = mujoco.MjData(model)

        # Set a visible "home" pose (slider midway, slight bends)
        home_qpos = np.zeros(model.nq)
        if model.nq >= 6:
            home_qpos[1] = config["slide_range_meters"]  # Set to 100% height
            home_qpos[2] = np.deg2rad(45)  # Bend 1
            home_qpos[3] = np.deg2rad(45)  # Bend 2
            home_qpos[5] = np.deg2rad(45)  # Wrist Bend

        draw_robot_wireframe(ax, model, data, qpos=home_qpos)
    except Exception as e:
        print(f"Warning: Could not draw robot wireframe. Make sure robot_config is accessible. Error: {e}")

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
    has_layers = "layer_label" in df.columns

    if is_monte_carlo and has_layers:
        # --- MODE 1: Stratified Spatial Point Cloud ---
        print(f"Rendering Stratified Point Cloud ({len(df)} points)...")

        # Color palette for the 5 layers (Plasma colormap creates nice thermal-like banding)
        layer_names = ["0%", "25%", "50%", "75%", "100%"]
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(layer_names)))

        for idx, layer_name in enumerate(layer_names):
            layer_df = hits.filter(pl.col("layer_label") == layer_name)
            # Draw the point cloud for this specific height layer
            ax.scatter(layer_df["x"], layer_df["y"], layer_df["z"],
                       color=colors[idx], s=2, alpha=0.3, label=f"Slider: {layer_name}")

        # Calculate absolute coverage score
        in_zone_count = hits.filter(pl.col("in_zone")).height
        coverage = (in_zone_count / len(df)) * 100

        print(f"--- ADL INTERSECTION SCORE ---")
        print(f"Total Configurations: {len(df)}")
        print(f"Zone A Hits: {in_zone_count}")
        print(f"Coverage Metric: {coverage:.2f}%")
        ax.set_title(f"6-DOF Stratified Reachability (Zone A Coverage: {coverage:.2f}%)")
        ax.legend(loc="upper right", title="Passive Slider Height")

    elif is_monte_carlo:
        # --- MODE 2: Standard Point Cloud (Fallback) ---
        print(f"Rendering Point Cloud ({len(df)} points)...")
        in_zone_df = hits.filter(pl.col("in_zone"))
        out_zone_df = hits.filter(~pl.col("in_zone"))

        ax.scatter(out_zone_df["x"], out_zone_df["y"], out_zone_df["z"], c='darkgray', s=2, alpha=0.3)
        ax.scatter(in_zone_df["x"], in_zone_df["y"], in_zone_df["z"], c='limegreen', s=10, alpha=0.8)

        coverage = (len(in_zone_df) / len(df)) * 100
        ax.set_title(f"Monte Carlo Reachability (Zone A Coverage: {coverage:.2f}%)")

    else:
        # --- MODE 3: Voxels (Low-Res Grid Search) ---
        print(f"Rendering Voxels ({len(df)} points)...")
        VOXEL_RADIUS = 0.05
        for row in hits.iter_rows(named=True):
            point = [row["x"], row["y"], row["z"]]
            color = 'limegreen' if row["in_zone"] else 'darkgray'
            draw_sphere(ax, point, VOXEL_RADIUS, color, alpha=0.4)
        ax.set_title("Discrete Workspace Voxelization")

    # --- Plot formatting ---
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
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

    plt.show()


if __name__ == "__main__":
    visualize_workspace()

