import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


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