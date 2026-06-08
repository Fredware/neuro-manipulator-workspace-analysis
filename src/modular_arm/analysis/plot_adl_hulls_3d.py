import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from modular_arm.core.adl_config import settings
from modular_arm.analysis.adl_envelope_generator import generate_cohort_hulls


def plot_comparative_hulls(adl_zones: dict):
    """
    Renders an interactive 3D Matplotlib plot overlaying the Healthy (HS)
    and Post-Stroke (ST) workspace boundaries for visual validation.
    """
    # Create a subplot for each clinical zone
    num_zones = len(adl_zones)
    fig = plt.figure(figsize=(6 * num_zones, 6))
    fig.canvas.manager.set_window_title('ADL Workspace Sanity Check')

    # Styling configuration
    cohort_styles = {
        "HS": {"color": "green", "alpha": 0.15, "label": "Healthy Baseline"},
        "ST": {"color": "red", "alpha": 0.4, "label": "Post-Stroke Impairment"}
    }

    for idx, (zone_name, cohorts) in enumerate(adl_zones.items(), 1):
        ax = fig.add_subplot(1, num_zones, idx, projection='3d')
        ax.set_title(f"Clinical {zone_name.replace('_', ' ')}")

        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])

        # Render each cohort's geometric volume
        for cohort_name in ["HS", "ST"]:
            if cohort_name not in cohorts:
                continue

            geom = cohorts[cohort_name]
            hull = geom["hull"]

            # SciPy's hull.simplices contains the indices of the points forming each triangular face
            faces = [hull.points[simplex] for simplex in hull.simplices]

            poly3d = Poly3DCollection(
                faces,
                alpha=cohort_styles[cohort_name]["alpha"],
                facecolor=cohort_styles[cohort_name]["color"],
                edgecolors='black',
                linewidths=0.2
            )
            ax.add_collection3d(poly3d)

            # Update axis limits for dynamic scaling
            global_min = np.minimum(global_min, geom["aabb_min"])
            global_max = np.maximum(global_max, geom["aabb_max"])

        # Enforce isotropic 1:1:1 aspect ratio so the geometries aren't distorted
        max_range = np.max(global_max - global_min) / 2.0
        mid_x = (global_max[0] + global_min[0]) / 2.0
        mid_y = (global_max[1] + global_min[1]) / 2.0
        mid_z = (global_max[2] + global_min[2]) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect([1, 1, 1])

        # Draw the sternum origin (0,0,0) as a reference point
        ax.scatter(0, 0, 0, color='black', s=50, label='Sternum Origin', zorder=5)

        ax.set_xlabel('X (m) - Forward')
        ax.set_ylabel('Y (m) - Vertical')
        ax.set_zlabel('Z (m) - Lateral')

        # Custom legend handling to avoid duplicate labels from Poly3DCollection
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Healthy Baseline (HS)'),
            Patch(facecolor='red', alpha=0.6, label='Post-Stroke (ST)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Sternum Origin')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    target_csv = Path(settings.adl_csv_path)

    print("🚀 Generating geometric volumes for visual validation...")
    try:
        # Generate the exact structures that will be passed to Monte Carlo
        clinical_zones = generate_cohort_hulls(target_csv, settings.zone_mappings)
        plot_comparative_hulls(clinical_zones)
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

