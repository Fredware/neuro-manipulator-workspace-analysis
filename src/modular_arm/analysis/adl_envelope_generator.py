import numpy as np
import polars as pl
import structlog
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay
from typing import Dict, Any, List

from modular_arm.core.adl_config import settings

logger = structlog.get_logger()

def normalize_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Applies anatomical normalization to convert global coordinates to local.
    :param df:
    :return:
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
    """Trim spatial outliers using double-tailed quantile to isolate core envelope

    :param df:
    :return: Nx3 numpy array formatted for scipy spatial module
    """
    bounds = df.select([
        pl.col("x_local").quantile(0.025).alias("x_min"),
        pl.col("x_local").quantile(0.975).alias("x_max"),
        pl.col("y_local").quantile(0.025).alias("y_min"),
        pl.col("y_local").quantile(0.975).alias("y_max"),
        pl.col("z_local").quantile(0.025).alias("z_min"),
        pl.col("z_local").quantile(0.975).alias("z_max"),
    ]).row(0)

    trimmed_df = df.filter(
        (pl.col("x_local").is_between(bounds[0], bounds[1])) &
        (pl.col("y_local").is_between(bounds[2], bounds[3])) &
        (pl.col("z_local").is_between(bounds[4], bounds[5]))
    )
    return trimmed_df.select(["x_local", "y_local", "z_local"]).to_numpy()

def generate_cohort_hulls(csv_path: Path, zone_mapping: dict[str, list[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Load CSV file containing all data, normalizes it, and constructs the HS, ST, and Combined spatial geometries.

    Axis-aligned bounding boxes (AABB) are generated for the broad-phase check
    Delaunay tessellations are generated for the narrow phase check

    :param csv_path:
    :param zone_mapping:
    :return: A nested dictionary containing the ConvexHull, Delaunay tessellation, and AABB boundaries for each zone and cohort
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} not found. Run the MATLAB script to generate it.")


    logger.info("Loading Kinematics CSV", file=csv_path.name)

    raw_df = pl.read_csv(csv_path)
    normalized_df = normalize_kinematics(raw_df)

    adl_envelopes = {}

    # Map and generate geometries
    for zone, tasks in zone_mapping.items():
        logger.info("Computing spatial geometries", zone=zone)
        zone_data = normalized_df.filter(pl.col("task_code").is_in(tasks))

        if zone_data.is_empty():
            logger.warning("No data found for mapped tasks", zone=zone)
            continue

        adl_envelopes[zone] = {}

        cohort_slices = {
            "HS": zone_data.filter(pl.col("cohort") == "HS"),
            "ST": zone_data.filter(pl.col("cohort") == "ST"),
            "ALL": zone_data
        }

        for cohort_name, slice_df in cohort_slices.items():
            if slice_df.is_empty():
                continue

            core_kinematics = filter_95th_percentile(slice_df)

            # SciPy Geometry Generation
            hull = ConvexHull(core_kinematics)
            hull_tessellation = Delaunay(core_kinematics[hull.vertices])
            aabb_min = np.min(core_kinematics[hull.vertices], axis=0)
            aabb_max = np.max(core_kinematics[hull.vertices], axis=0)

            adl_envelopes[zone][cohort_name] = {
                "hull": hull,
                "tessellation": hull_tessellation,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
            }

            logger.debug("Hull generated", zone=zone, cohort=cohort_name, volume=np.round(hull.volume, 4))

    return adl_envelopes

if __name__ == "__main__":
    target_csv = Path(settings.adl_csv_path)

    try:
        adl_zones = generate_cohort_hulls(target_csv, settings.zone_mappings)

        print("\n=== Geometric Volume Analysis (m^3) ===")

        for zone, cohorts in adl_zones.items():
            print(f"\n[{zone}]")
            for cohort, geom in cohorts.items():
                print(f"\t{cohort} Hull Volume: {geom['hull'].volume:.5f} m^3")
    except Exception as e:
        logger.exception("Pipeline failed", error=str(e))

