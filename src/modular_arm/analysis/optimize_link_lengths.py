"""Optimize ARM link lengths for ADL workspace coverage.

Searches over [dof2_start_height, dof2_slide_range, dof3_length, dof4_length,
dof6_length] using differential evolution. dof1_height is derived (phantom variable).
EE length is fixed.

Usage:
    uv run python -m modular_arm.analysis.optimize_link_lengths
    uv run python -m modular_arm.analysis.optimize_link_lengths --evaluate-nominal
    uv run python -m modular_arm.analysis.optimize_link_lengths --report-reach
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.optimize
import structlog

from modular_arm.analysis.workspace_evaluator import (
    CoverageEvaluator,
    build_zone_targets,
    extract_scene_anchors,
    required_reach_per_zone,
)
from modular_arm.core.config import get_optimizer_settings, get_paths
from modular_arm.core.frames import validate_transform

logger = structlog.get_logger(__name__)

# --- Design vector layout ---
LENGTH_NAMES = (
    "dof2_start_height",
    "dof2_slide_range",
    "dof3_length",
    "dof4_length",
    "dof6_length",
)

# Anthropometric seed (Winter stature ratios, ~1.75m male) [meters]
ANTHROPOMETRIC_SEED_M = np.array([0.07, 0.18, 0.33, 0.26, 0.09])

# Search bounds [meters] — (lo, hi) per variable
# TODO: tighten from CAD measurements
DEFAULT_BOUNDS_M = [
    (0.04, 0.12),   # dof2_start_height
    (0.05, 0.30),   # dof2_slide_range
    (0.20, 0.40),   # dof3_length
    (0.15, 0.35),   # dof4_length
    (0.05, 0.15),   # dof6_length
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize ARM link lengths for ADL coverage.")
    parser.add_argument("--arm-xml", type=Path, default=None,
                        help="Nominal ARM MJCF for anchor extraction.")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("data/geometry-optimization/best.json"))
    parser.add_argument("--evaluate-nominal", action="store_true",
                        help="Score the anthropometric seed and exit.")
    parser.add_argument("--report-reach", action="store_true",
                        help="Print required reach per zone and exit.")
    args = parser.parse_args()

    # --- Validate invariants ---
    validate_transform()

    # --- One-time setup (expensive, happens once) ---
    arm_xml = args.arm_xml or (get_paths().robot_mjcf_dir / "neuro_arm.xml")
    anchors = extract_scene_anchors(arm_xml)   # sternum_pos, base_pos, base_mat (numpy arrays)
    zone_targets = build_zone_targets()         # dict[zone_name -> (N,3) numpy arrays in lab frame]

    # --- Report-only modes ---
    if args.report_reach:
        reaches = required_reach_per_zone(zone_targets)
        nominal_reach = ANTHROPOMETRIC_SEED_M[2:].sum()  # dof3 + dof4 + dof6
        print(f"\nNominal straight-line reach (DOF3+DOF4+DOF6): {nominal_reach:.3f} m")
        for zone, r in reaches.items():
            gap = r - nominal_reach
            flag = "  <-- SHORT" if gap > 0 else ""
            print(f"  {zone}: requires {r:.3f} m  (margin {-gap:+.3f} m){flag}")
        return 0

    # --- Build the evaluator (picklable: only numpy + scipy state) ---
    settings = get_optimizer_settings()
    evaluator = CoverageEvaluator(
        anchors=anchors,
        zone_targets=zone_targets,
        n_samples=settings.n_joint_samples,
        eps=settings.reach_eps_m,
        seed=args.seed,
    )

    if args.evaluate_nominal:
        scalar, per_zone = evaluator.score(ANTHROPOMETRIC_SEED_M)
        print(f"\nAnthropometric seed coverage: {scalar:.4f}")
        for z, v in per_zone.items():
            print(f"  {z}: {v:.4f}")
        return 0

    # --- Run DE ---
    x0 = np.clip(ANTHROPOMETRIC_SEED_M,
                  [b[0] for b in DEFAULT_BOUNDS_M],
                  [b[1] for b in DEFAULT_BOUNDS_M])

    logger.info("optimization_start", maxiter=args.maxiter, popsize=args.popsize)

    result = scipy.optimize.differential_evolution(
        func=evaluator,                  # evaluator.__call__(x) -> float (negative coverage)
        bounds=DEFAULT_BOUNDS_M,
        x0=x0,
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        init="sobol",
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-3,
        polish=False,                    # L-BFGS polish is meaningless on count-based objectives
        workers=-1,                      # multiprocessing (evaluator must be picklable)
        updating="deferred",             # required when workers > 1
        disp=True,
    )

    # --- Report and save ---
    best_lengths = dict(zip(LENGTH_NAMES, result.x))
    best_scalar, best_per_zone = evaluator.score(result.x)

    logger.info("optimization_done", coverage=round(best_scalar, 4),
                per_zone={k: round(v, 4) for k, v in best_per_zone.items()})

    output = {
        "best_lengths_m": best_lengths,
        "coverage_scalar": best_scalar,
        "coverage_per_zone": best_per_zone,
        "n_evals": int(result.nfev),
        "success": bool(result.success),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    logger.info("result_saved", path=str(args.out))

    print("\n=== Best geometry (m) ===")
    for name, val in best_lengths.items():
        print(f"  {name}: {val:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())