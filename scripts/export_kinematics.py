#!/usr/bin/env python
"""Export kinematic marker data from the Figshare upper-limb .mat files.

Replaces the original MATLAB export_kinematics.m with a pure-Python pipeline so the
build no longer depends on a MATLAB license.

The .mat files store a struct ``s`` whose layout is documented in Lucchetti et al.
(2025), Scientific Data 12:1904, Table 2 and Figure 1a:

    s.DataULdom   — healthy subjects (dominant side)
    s.DataULpleg  — stroke subjects (plegic side)

Each field is a 1x6 struct array (one per task) with fields:
    .TaskCode        — BA, BC, HH, HM, PS, SC (fixed order)
    .MarkerVarName   — 19 marker names (fixed protocol)
    .Marker          — (3*Nmarker, N_frames) float matrix, units: meters

scipy.io.loadmat reads the numeric Marker matrices correctly but cannot deserialize
MATLAB's newer string arrays (they come back as opaque MCOS references). Since the
marker order and task codes are standardized by the published protocol, we hardcode
them from the paper rather than parsing the garbled strings.

Marker protocol (Figure 1a, Table 2):
    0  C7               Midline
    1  MANUBRIUM         Midline (upper sternum / suprasternal notch)
    2  SHOULDER_RX       Bilateral
    3  SHOULDER_LX       Bilateral
    4  MIDARM_{side}     Lateralized
    5  ELBOW_{side}      Lateralized
    6  MIDFORE_{side}    Lateralized
    7  ULNA_{side}       Lateralized
    8  RADIUS_{side}     Lateralized
    9  Meta1_{side}      Hand — thumb MCP
    10 Pro1_{side}       Hand — thumb PIP
    11 Dis1_{side}       Hand — thumb DIP
    12 Meta2_{side}      Hand — index MCP  <-- "hand" reference
    13 Pro2_{side}       Hand — index PIP
    14 Dis2_{side}       Hand — index DIP
    15 Meta5_{side}      Hand — little MCP
    16 Pro5_{side}       Hand — little PIP
    17 Dis5_{side}       Hand — little DIP
    18 OBJECT/FOREHEAD   Task-dependent (absent in PS -> only 18 markers)

Downstream contract (adl_envelope_generator.normalize_kinematics):
    Columns: hand_x, hand_y, hand_z, sternum_x, sternum_y, sternum_z  — millimeters
             task_code, cohort, subject

Usage:
    uv run python scripts/export_kinematics.py --check data/kinematic-emg-adl-dataset/HS01.mat
    uv run python scripts/export_kinematics.py data/kinematic-emg-adl-dataset/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.io
import structlog

logger = structlog.get_logger(__name__)

# --- Canonical schema from the paper (Lucchetti et al. 2025, Sci Data 12:1904) ---

TASK_CODES = ("BA", "BC", "HH", "HM", "PS", "SC")
"""Fixed task order in every .mat file (6 tasks)."""

# 0-based marker indices into the (3*N_markers, N_frames) Marker matrix.
# Each marker occupies 3 consecutive rows: [x, y, z].
MANUBRIUM_IDX = 1   # sternum / suprasternal notch (midline, no side suffix)
META2_IDX = 12       # 2nd metacarpophalangeal joint — standard "hand" reference point

# Paper Table 2 says Marker units are meters. Downstream normalize_kinematics divides
# by 1e3 (expecting millimeters). This scale factor bridges the two; set to 1.0 if the
# downstream code is changed to expect meters.
M_TO_MM = 1.0


def _extract_xyz(marker_matrix: np.ndarray, marker_idx: int) -> np.ndarray:
    """Extract (N_frames, 3) from the (3*N_markers, N_frames) Marker matrix."""
    row = marker_idx * 3
    return marker_matrix[row : row + 3, :].T


def _load_mat(path: Path) -> dict:
    """Load a .mat file with scipy, squeezing and unpacking structs."""
    return scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _get_tasks(mat: dict) -> np.ndarray:
    """Return the task struct array for the primary limb."""
    s = mat["s"]
    if hasattr(s, "DataULdom"):
        tasks = s.DataULdom
    elif hasattr(s, "DataULpleg"):
        tasks = s.DataULpleg
    else:
        raise KeyError("Struct 's' has neither 'DataULdom' nor 'DataULpleg'")
    # Single-task edge case: ensure iterable
    if not hasattr(tasks, "__len__"):
        tasks = np.array([tasks])
    return tasks


# --- Check mode: print sample values to verify units + marker indices ---

def check(mat_path: Path) -> None:
    """Print sample marker values to verify units and index correctness."""
    mat = _load_mat(mat_path)
    tasks = _get_tasks(mat)
    subject = mat_path.stem
    side_field = "DomSide" if hasattr(mat["s"], "DomSide") else "HemiSide"
    side = str(getattr(mat["s"], side_field))

    print(f"\n=== Unit check: {subject} (side={side}) ===")
    print(f"Tasks found: {len(tasks)}  (expected 6)")
    print(f"Task codes (hardcoded): {TASK_CODES}")

    for i, task in enumerate(tasks):
        code = TASK_CODES[i] if i < len(TASK_CODES) else f"?{i}"
        marker = task.Marker
        n_markers = marker.shape[0] // 3

        manub = _extract_xyz(marker, MANUBRIUM_IDX)
        meta2 = _extract_xyz(marker, META2_IDX)

        print(f"\n  Task {code}: Marker shape {marker.shape} ({n_markers} markers x {marker.shape[1]} frames)")
        print(f"    MANUBRIUM (idx {MANUBRIUM_IDX}) frame 0: {manub[0].round(4)}")
        print(f"    Meta2     (idx {META2_IDX}) frame 0: {meta2[0].round(4)}")

        # If values are ~0.1-2.0, data is in meters (paper's stated unit).
        # If values are ~100-2000, data is in millimeters.
        max_val = max(np.abs(manub[0]).max(), np.abs(meta2[0]).max())
        if max_val < 10:
            print(f"    -> Values look like METERS (max={max_val:.3f}). M_TO_MM={M_TO_MM} will convert to mm.")
        else:
            print(f"    -> Values look like MILLIMETERS already (max={max_val:.1f}). Set M_TO_MM=1.0.")


# --- Export mode ---

def export_all(mat_dir: Path, output_csv: Path) -> None:
    """Extract hand + sternum markers from all .mat files and write the CSV."""
    mat_files = sorted(p for p in mat_dir.glob("*.mat") if p.stem[:2] in ("HS", "ST"))
    if not mat_files:
        print(f"No HS*.mat or ST*.mat files found in {mat_dir}")
        sys.exit(1)

    all_rows: list[dict] = []
    skipped: list[str] = []

    for mat_path in mat_files:
        subject = mat_path.stem
        cohort = "HS" if subject.startswith("HS") else "ST"
        logger.info("processing", subject=subject, cohort=cohort)

        try:
            mat = _load_mat(mat_path)
            tasks = _get_tasks(mat)
        except Exception as e:
            logger.error("load_failed", subject=subject, error=str(e))
            skipped.append(subject)
            continue

        if len(tasks) != len(TASK_CODES):
            logger.warning("task_count_mismatch", subject=subject,
                           expected=len(TASK_CODES), got=len(tasks))

        for i, task in enumerate(tasks):
            if i >= len(TASK_CODES):
                break
            code = TASK_CODES[i]
            marker = task.Marker
            n_markers = marker.shape[0] // 3

            if META2_IDX >= n_markers:
                logger.warning("marker_missing", subject=subject, task=code,
                               n_markers=n_markers, need=META2_IDX)
                continue

            manub = _extract_xyz(marker, MANUBRIUM_IDX) * M_TO_MM
            meta2 = _extract_xyz(marker, META2_IDX) * M_TO_MM
            n_frames = manub.shape[0]

            for f in range(n_frames):
                all_rows.append({
                    "hand_x": float(meta2[f, 0]),
                    "hand_y": float(meta2[f, 1]),
                    "hand_z": float(meta2[f, 2]),
                    "sternum_x": float(manub[f, 0]),
                    "sternum_y": float(manub[f, 1]),
                    "sternum_z": float(manub[f, 2]),
                    "task_code": code,
                    "cohort": cohort,
                    "subject": subject,
                })

        logger.info("subject_done", subject=subject, rows_so_far=len(all_rows))

    if not all_rows:
        print("No data extracted. Run --check on a .mat file to diagnose.")
        sys.exit(1)

    df = pl.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_csv)

    print(f"\n=== Export complete ===")
    print(f"Subjects: {len(mat_files) - len(skipped)} processed, {len(skipped)} skipped")
    print(f"Total frames: {len(df):,}")
    print(f"Tasks: {sorted(df['task_code'].unique().to_list())}")
    print(f"Cohorts: {sorted(df['cohort'].unique().to_list())}")
    print(f"Saved to: {output_csv}")
    if skipped:
        print(f"Skipped: {skipped}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export kinematic marker data from Figshare .mat files (no MATLAB needed)."
    )
    parser.add_argument(
        "path", type=Path,
        help="Single .mat file (with --check) or directory of .mat files (export mode).",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("data/kinematic-emg-adl-dataset/clean_adl_kinematics.csv"),
        help="Output CSV path (export mode).",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Print sample marker values from one .mat file to verify units and indices.",
    )
    args = parser.parse_args()

    if args.check:
        if not args.path.is_file():
            print(f"--check requires a single .mat file, got: {args.path}")
            return 1
        check(args.path)
        return 0

    if args.path.is_file():
        print("For single-file inspection use --check. For export pass a directory.")
        return 1

    export_all(args.path, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

