"""Generate ADL Envelopes from empirical kinematic data.

Pipeline:
    1. Load raw CSV file containing all transformed .mat data
    2. Normalize kinematics to local coordinates (sternum - hand)
    3. Trim outliers using 95th percentile threshold
    4. Generate ConvexHull and Delaunay tessellations per zone per cohort
    5. Compute AABB boundaries

Output modes:
    In-memory: generate_cohort_hulls() returns nested dicts of scipy geom objects for use by the optimizer and plotter.
    STL export: export_hulls_as_stl() writes each hull as an STL for renderingin MuJoCo alongside the wheelchair/human
        scene. Hulls are in sternum-relative coordinates, so they render correctly when parented to a body at the
        sternum site.

Usage:
    uv run python -m modular_arm.analysis.adl_envelope_generator
    uv run python -m modular_arm.analysis.adl_envelope_generator --export-stl
    uv run python -m modular_arm.analysis.adl_envelope_generator --export-stl --stl-dir data/custom/
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
from scipy.spatial import ConvexHull, Delaunay

from modular_arm.core.config import settings

logger = structlog.get_logger(__name__)

# --- Default Output Paths --- TODO(@FRM): move to config.yaml
DEFAULT_STL_DIR = Path("data/adl-envelopes")

# --- Zone Colors for MuJoCo rendering --- TODO(@FRM): move to config.yaml
# RGBA with low alpha so that the ARM is visible through the hulls
# Shared with assemble_scene.py via import
ADL_ZONE_COLORS: dict[str, list[float]] = {
    "zone_a": [0.90, 0.30, 0.20, 0.25], # warm red: personal care (HM, HH)
    "zone_b": [0.20, 0.80, 0.30, 0.25], # green: grasping (BA, BC, SC)
    "zone_c": [0.25, 0.45, 0.90, 0.25], # blue: prono-supination (PS)
}

DEFAULT_COHORT = "ALL"
"""Which cohort to export by default. ALL = healthy + stroke combined."""

# --- Normalization and filtering ---
def normalize_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Applies anatomical normalization to convert global coordinates to local.

    Args:
        df: Raw kinematic dataframe with hand_x/y/z and sternum_x/y/z columns in millimeters.

    Returns:
        Dataframe with added x_local, y_local, z_local columns in meters, with NaN/null rows removed.
    """
    normalized =  df.with_columns([
        ((pl.col("hand_x") - pl.col("sternum_x"))/1e3).alias("x_local"),
        ((pl.col("hand_y") - pl.col("sternum_y"))/1e3).alias("y_local"),
        ((pl.col("hand_z") - pl.col("sternum_z"))/1e3).alias("z_local"),
    ])

    return normalized.drop_nulls(subset=["x_local", "y_local", "z_local"]).filter(
        pl.col("x_local").is_not_nan() &
        pl.col("y_local").is_not_nan() &
        pl.col("z_local").is_not_nan()
    )

def filter_95th_percentile(df: pl.DataFrame) -> np.ndarray:
    """Trim spatial outliers using double-tailed 2.5-97.5 quantiles to isolate core envelopes.

    Args:
        df: Dataframe with x_local, y_local, z_local columns in meters.

    Returns:
        Nx3 numpy array of trimmed kinematics for scipy spatial ops
    """
    quant_lo = 0.025
    quant_hi = 0.975
    bounds = df.select([
        pl.col("x_local").quantile(quant_lo).alias("x_min"),
        pl.col("x_local").quantile(quant_hi).alias("x_max"),
        pl.col("y_local").quantile(quant_lo).alias("y_min"),
        pl.col("y_local").quantile(quant_hi).alias("y_max"),
        pl.col("z_local").quantile(quant_lo).alias("z_min"),
        pl.col("z_local").quantile(quant_hi).alias("z_max"),
    ]).row(0)

    trimmed_df = df.filter(
        (pl.col("x_local").is_between(bounds[0], bounds[1])) &
        (pl.col("y_local").is_between(bounds[2], bounds[3])) &
        (pl.col("z_local").is_between(bounds[4], bounds[5]))
    )
    return trimmed_df.select(["x_local", "y_local", "z_local"]).to_numpy()

# --- Hull Generation ---

def generate_cohort_hulls(
        csv_path: Path,
        zone_mapping: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Generate convex hulls and Delaunay tessellations per zone per cohort.

    For each zone, three cohorts are produced: HS (healthy), ST (stroke), ALL (combined).
    Each cohort contains a ConvexHull, Delaunay tessellation, and axis-aligned bounding box (AABB) for the two-phase
    broad/narrow containment test used by the coverage optimizer.

    Args:
        csv_path: Path to the kinematic CSV with hand / sternum columns.
        zone_mapping: Dict mapping zone names to task codes.

    Returns:
        Nested dict: envelopes[zone][cohort] = {hull, tessellation, aabb_min, aabb_max}
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"File {csv_path} not found. Run the MATLAB script to generate it."
        )

    logger.info("Loading Kinematics CSV", file=csv_path.name)
    raw_df = pl.read_csv(csv_path)
    normalized_df = normalize_kinematics(raw_df)

    adl_envelopes: dict[str, dict[str, Any]] = {}

    # Map and generate geometries
    for zone, tasks in zone_mapping.items():
        logger.info("computing_spatial_geometries", zone=zone)
        zone_data = normalized_df.filter(pl.col("task_code").is_in(tasks))

        if zone_data.is_empty():
            logger.warning("no_data_for_zone", zone=zone)
            continue

        adl_envelopes[zone] = {}

        cohort_slices = {
            "HS": zone_data.filter(pl.col("cohort") == "HS"),
            "ST": zone_data.filter(pl.col("cohort") == "ST"),
            "ALL": zone_data,
        }

        for cohort_name, slice_df in cohort_slices.items():
            if slice_df.is_empty():
                continue

            core_kinematics = filter_95th_percentile(slice_df)
            # SciPy Geometry Generation
            hull = ConvexHull(core_kinematics)
            hull_vertices = core_kinematics[hull.vertices]
            tessellation = Delaunay(hull_vertices)
            aabb_min = np.min(hull_vertices, axis=0)
            aabb_max = np.max(hull_vertices, axis=0)

            adl_envelopes[zone][cohort_name] = {
                "hull": hull,
                "tessellation": tessellation,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
            }

            logger.debug(
                "hull_generated",
                zone=zone, cohort=cohort_name,
                n_vertices=len(hull.vertices),
                volume=np.round(hull.volume, 4),
            )

    return adl_envelopes

# --- STL Export ---
# Coordinate frame mapping
# The optoelectronic syste (BTS SMART-DX EVO) uses:
#   Lab +X = lateral (subject's left)
#   Lab +Y = vertical (up)
#   Lab +Z = behind the subject (away from the table)
#
# MuJoCo / ada_assets convention:
#    MuJoCo X = forward
#    MuJoCo Y = lateral
#    MuJoCo Z = up
#
# Transform: swap Y and Z columns
LAB_TO_MUJOCO = np.array([
    [0, 0, -1], # MuJoCo X (forward) = - Lab Z
    [-1, 0, 0], # MuJoCo Y (lateral) = - Lab X
    [0, 1, 0],  # MuJoCo Z (up)      = + Lab Y
], dtype=np.float64)

def hull_to_binary_stl(
        hull: ConvexHull,
        path: Path,
        *,
        transform: np.ndarray | None = None,
) -> None:
    """Write a ConvexHull object to a binary STL file.

    The hull's simplices (triangular faces) and equations (outward normals) are written directly. The resulting STL is
    in the same coordinate frame as the hull points - for ADL envelopes, this is the sternum-relative frame.

    Args:
        hull: Scipy ConvexHull object with aligned simplices and equations.
        path: Output path for the STL file.
        transform: Optional transformation matrix applied to both vertices and normals before writing. Use LAB_TO_MUJOCO
            when exporting for MuJoCo VIS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_faces = len(hull.simplices)

    # Pre-transform all points and normals if needed
    points = hull.points
    normals = hull.equations[:, :3]
    if transform is not None:
        points = points @ transform.T
        normals = normals @ transform.T

    with open(path, "wb") as f:
        # 80-byte header
        header = f"ADL hull | {n_faces} faces".encode("ascii")
        f.write(header.ljust(80, b"\x00"))
        # Triangle count
        f.write(struct.pack("<I", n_faces))

        for i, simplex in enumerate(hull.simplices):
            # Outward normal from hull equations [a, b, c, d]
            f.write(struct.pack("<3f", *normals[i]))
            # Three vertices
            for vertex_idx in simplex:
                f.write(struct.pack("<3f", *points[vertex_idx]))
            # Attribute byte count (unused)
            f.write(struct.pack("<H", 0))
    logger.debug("stl_written", file=str(path), n_faces=n_faces)

def export_hulls_as_stl(
        adl_envelopes: dict[str, dict[str, Any]],
        output_dir: Path = DEFAULT_STL_DIR,
        cohort: str = DEFAULT_COHORT,
) -> list[Path]:
    """Export ADL convex hulls as binary STL files for MuJoCo rendering.

    File naming convention: {zone}_{cohort}.stl, e.g. zone_a_HS.stl.
    The files are loaded by assemble_scene.py --with-adl-hulls

    Args:
        adl_envelopes: Output of generate_cohort_hulls()
        output_dir: Directory to write STL files.
        cohort: Which cohort to export ("HS", "ST", or "ALL")

    Returns:
        List of paths to the written STL files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for zone_name, cohorts in adl_envelopes.items():
        if cohort not in cohorts:
            logger.warning("cohort_not_found", cohort=cohort, zone=zone_name, available=list(cohorts.keys()))
            continue

        hull = cohorts[cohort]["hull"]
        stl_path = output_dir / f"{zone_name}_{cohort}.stl"
        hull_to_binary_stl(hull, stl_path, transform=LAB_TO_MUJOCO)
        written.append(stl_path)
        logger.info(
            "hull_export_complete",
            zone=zone_name, cohort=cohort,
            path=str(stl_path),
            volume=np.round(hull.volume, 4),
        )

    return written

# --- CLI Entry Point ---
def main() -> int:
    """Generate ADL envelopes and optionally export as STL"""
    parser = argparse.ArgumentParser(
        description="Generate ADL workspace envelopes from kinematic data"
    )
    parser.add_argument(
        "--export-stl", action="store_true",
        help="Export ConvexHulls as STL files for MuJoCo rendering"
    )
    parser.add_argument(
        "--stl-dir", type=Path, default=DEFAULT_STL_DIR,
        help=f"Directory to write STL files (default: {DEFAULT_STL_DIR})"
    )
    parser.add_argument(
        "--cohort", choices=["HS", "ST", "ALL"], default=DEFAULT_COHORT,
        help=f"Cohort to export (default: {DEFAULT_COHORT})"
    )
    args = parser.parse_args()

    target_csv = Path(settings.adl_csv_path)

    try:
        adl_zones = generate_cohort_hulls(target_csv, settings.zone_mappings)
    except Exception as e:
        logger.exception("Pipeline failed", error=str(e))
        return 1

    # Volume summary
    print("\n=== Geometric Volume Analysis (m^3) ===")
    for zone, cohorts in adl_zones.items():
        print(f"\n[{zone}]")
        for cohort_name, geom in cohorts.items():
            print(f"\t{cohort_name} Hull Volume: {geom['hull'].volume:.4f} m^3")

    if args.export_stl:
        written = export_hulls_as_stl(
            adl_zones, output_dir=args.stl_dir, cohort=args.cohort,
        )
        print(f"Exported {len(written)} STL files to {args.stl_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

