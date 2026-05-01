import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_envelope():
    # 1. Load the data using Polars
    file_path = os.path.join("data", "workspace_envelope.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run the sweeper first!")
        return

    # Eager read
    df = pl.read_csv(file_path)

    # 2. Setup 3D Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Extract columns as numpy (for Matplotlib compatibility)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    # 4. Plot the workspace points
    # Color by height (z) to help with 3D depth perception
    img = ax.scatter(x, y, z, c=z, cmap='plasma', s=1, alpha=0.4)
    fig.colorbar(img, ax=ax, label='Height (z) [m]', shrink=0.6)

    # 5. Define ADL Zone A (Personal Care)
    # Coordinates in meters: [min, max]
    zone_a = {'x': [0.1, 0.4], 'y': [-0.2, 0.2], 'z': [0.1, 0.4]}

    def draw_box(ax, bounds, color='cyan', label='Zone A'):
        xb, yb, zb = bounds['x'], bounds['y'], bounds['z']
        # Draw the 12 edges of the box
        for i in range(2):
            for j in range(2):
                ax.plot([xb[i], xb[i]], [yb[j], yb[j]], zb, color=color, linewidth=2)
                ax.plot([xb[i], xb[i]], yb, [zb[j], zb[j]], color=color, linewidth=2)
                ax.plot(xb, [yb[i], yb[i]], [zb[j], zb[j]], color=color, linewidth=2)

    draw_box(ax, zone_a)

    # 6. Formatting
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Lateral)')
    ax.set_zlabel('Z (Vertical)')
    ax.set_title('Neuro-Manipulator Workspace Envelope (Polars/MuJoCo)')

    # Force equal aspect ratio so the robot's reach doesn't look stretched
    ax.set_box_aspect([1, 1, 1])

    # Set plot limits based on data + buffer
    margin = 0.1
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(0, z.max() + margin)

    print(f"Displaying {len(df)} points...")
    plt.show()


if __name__ == "__main__":
    visualize_envelope()