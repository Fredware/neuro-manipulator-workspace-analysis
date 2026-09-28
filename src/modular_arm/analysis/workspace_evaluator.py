"""Evaluate a robot configuration's ADL workspace coverage.

Contains the core evaluation pipeline: FK sweep, coordinate bridge, and
coverage scoring. Designed to be called thousands of times by the optimizer
in ``optimize_link_lengths.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import structlog
from scipy.spatial import cKDTree
from scipy.stats import qmc

from modular_arm.analysis.adl_envelope_generator import generate_cohort_hulls
from modular_arm.core.config import get_adl_settings, get_paths
from modular_arm.core.frames import mujoco_to_lab, validate_transform
from modular_arm.robot.robot_config import build_robot_config
from modular_arm.scene.assemble_scene import build_composite_spec

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Scene anchors — extracted once, reused every evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneAnchors:
    """Floor-frame positions cached from the full wheelchair + human scene.

    These don't change with link lengths (the arm mounts at a fixed site on
    the wheelchair, the sternum is on the human). Extracting them once avoids
    recompiling the expensive full scene on every evaluation.

    Attributes:
        sternum_pos: ``human/sternum`` site position in floor frame [m].
        base_pos: Arm mount site position in floor frame [m].
        base_mat: Arm mount site 3x3 rotation in floor frame.
    """

    sternum_pos: np.ndarray  # (3,)
    base_pos: np.ndarray     # (3,)
    base_mat: np.ndarray     # (3, 3)


def extract_scene_anchors(arm_xml: Path) -> SceneAnchors:
    """Compile the full scene once and cache the two anchor poses.

    Any valid arm geometry works here — the mount pose and sternum are
    independent of link lengths.

    Args:
        arm_xml: Path to any valid arm component MJCF (used only to make
            the scene compile; the specific geometry doesn't matter).

    Returns:
        SceneAnchors with sternum, base position and orientation.
    """
    # Step 1: build the composite spec (wheelchair + arm + human)
    # Step 2: compile to MjModel + MjData
    # Step 3: run mj_forward to populate site positions
    # Step 4: read sternum site pos, base mount site pos and mat
    # Step 5: return SceneAnchors (pure numpy — picklable)


# ---------------------------------------------------------------------------
# ADL zone targets — voxelized hull interiors in lab frame
# ---------------------------------------------------------------------------
def build_zone_targets(
    pitch: float = 0.03,
    cohort: str = "ALL",
) -> dict[str, np.ndarray]:
    """Voxelize each ADL hull interior into target points in the lab frame.

    Coverage is measured as the fraction of these voxels reachable by the arm.
    This can't be gamed by shrinking the workspace (unlike counting EE samples
    that land inside a hull).

    Args:
        pitch: Grid spacing in meters. Keep roughly equal to the reach epsilon.
        cohort: Which cohort's hulls to use (HS, ST, or ALL).

    Returns:
        Dict mapping zone name to (N, 3) arrays of interior points in lab frame.
    """
    # Step 1: call generate_cohort_hulls() to get the in-memory tessellations
    # Step 2: for each zone, grid the AABB at `pitch` spacing
    # Step 3: keep only points where tessellation.find_simplex(point) >= 0
    # Step 4: return {zone_name: interior_points_array}


def required_reach_per_zone(zone_targets: dict[str, np.ndarray]) -> dict[str, float]:
    """Max distance from sternum (origin) to any voxel in each zone.

    This quantifies what each zone *demands* in terms of reach, so you can
    compare it against a candidate geometry's straight-line reach
    (dof3 + dof4 + dof6 + ee). If the required reach exceeds the arm's
    reach, that zone is geometrically infeasible — no optimizer can fix it.

    Args:
        zone_targets: Output of build_zone_targets().

    Returns:
        Dict mapping zone name to max reach distance in meters.
    """
    # Voxels are sternum-relative (origin = sternum), so distance is just norm
    return {z: float(np.linalg.norm(pts, axis=1).max()) for z, pts in zone_targets.items()}


# ---------------------------------------------------------------------------
# Coordinate bridge — arm-alone EE positions → lab frame
# ---------------------------------------------------------------------------
def ee_to_lab_frame(
    ee_positions: np.ndarray,
    anchors: SceneAnchors,
    arm_base_pos: np.ndarray,
    arm_base_mat: np.ndarray,
) -> np.ndarray:
    """Transform EE positions from arm-alone world frame to sternum-relative lab frame.

    Chain:
        arm-alone world → base-relative → base frame → floor frame
        → sternum-relative (MuJoCo axes) → lab frame

    Args:
        ee_positions: (N, 3) EE positions in arm-alone world coords.
        anchors: Cached floor-frame anchors from extract_scene_anchors().
        arm_base_pos: Base mount site position in the arm-alone model.
        arm_base_mat: Base mount site 3x3 rotation in the arm-alone model.

    Returns:
        (N, 3) EE positions in sternum-relative lab frame.
    """
    # Step 1: subtract arm_base_pos (arm-alone world → base-relative)
    # Step 2: rotate by arm_base_mat (base-relative → base frame)
    # Step 3: rotate by anchors.base_mat.T + translate by anchors.base_pos (→ floor frame)
    # Step 4: subtract anchors.sternum_pos (→ sternum-relative, MuJoCo axes)
    # Step 5: call mujoco_to_lab() (→ lab frame)


# ---------------------------------------------------------------------------
# Coverage evaluator — the optimizer's callable
# ---------------------------------------------------------------------------
class CoverageEvaluator:
    """Score a candidate geometry's ADL workspace coverage.

    Picklable: holds only numpy arrays and scipy objects (no MuJoCo state).
    MuJoCo models are built fresh inside each evaluation so the evaluator
    can be fanned across processes by ``differential_evolution(workers=-1)``.

    Shared state (set once in __init__, reused every evaluation):
        - scene anchors (sternum pos, base pose)
        - zone target voxels
        - fixed Sobol joint sample set (normalized to [0,1])
        - reach epsilon for KD-tree scoring

    Per-evaluation work (inside score / __call__):
        - build arm MuJoCo model from candidate lengths
        - FK sweep over fixed joint samples
        - coordinate bridge to lab frame
        - KD-tree epsilon query against zone voxels
    """

    def __init__(
        self,
        anchors: SceneAnchors,
        zone_targets: dict[str, np.ndarray],
        n_samples: int = 4096,
        eps: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.anchors = anchors
        self.zone_targets = zone_targets
        self.eps = eps
        # Fixed Sobol samples in [0,1]^6 — denormalized per candidate
        # against that candidate's actual joint ranges
        self._unit_samples = qmc.Sobol(d=6, scramble=True, seed=seed).random(n_samples)

    def _fk_sweep(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Run FK over the fixed joint sample set, return (N, 3) EE positions.

        For each sample:
            1. Denormalize from [0,1] to [joint_min, joint_max]
            2. Set qpos
            3. Call mj_kinematics (NOT mj_step — no contacts needed)
            4. Read ee_site position
        """
        # Step 1: read joint ranges from the model
        # Step 2: denormalize unit_samples to actual joint ranges
        # Step 3: for each sample, set qpos, call mj_kinematics, collect ee_site
        # Step 4: return (N, 3) array

    def _coverage(self, ee_lab: np.ndarray) -> dict[str, float]:
        """Per-zone volumetric reachability via KD-tree epsilon query.

        Args:
            ee_lab: (N, 3) EE positions in sternum-relative lab frame.

        Returns:
            Dict mapping zone name to fraction of voxels reached (0.0 to 1.0).
        """
        # Step 1: build KD-tree from ee_lab
        # Step 2: for each zone, query: how many voxels have a neighbor within eps?
        # Step 3: coverage = reached_voxels / total_voxels

    def score(self, x: np.ndarray) -> tuple[float, dict[str, float]]:
        """Evaluate one candidate geometry.

        Args:
            x: Design vector [dof2_start_height, dof2_slide_range,
               dof3_length, dof4_length, dof6_length] in meters.

        Returns:
            (scalar_coverage, per_zone_dict) where scalar is the mean
            coverage across zones and per_zone_dict maps zone names to
            individual coverage fractions.
        """
        # Step 1: pack x into a lengths dict for build_robot_config
        # Step 2: build_robot_config(lengths_m) → get assembler
        # Step 3: generate MJCF XML string, compile to MjModel + MjData
        # Step 4: FK sweep → (N, 3) EE positions in arm-alone frame
        # Step 5: read base mount site pose from the arm-alone model
        # Step 6: ee_to_lab_frame() → (N, 3) in sternum-relative lab frame
        # Step 7: self._coverage() → per-zone scores
        # Step 8: reduce to scalar (mean across zones)
        # Step 9: return (scalar, per_zone_dict)

    def __call__(self, x: np.ndarray) -> float:
        """DE-compatible callable. Returns NEGATIVE coverage (DE minimizes)."""
        try:
            scalar, _ = self.score(x)
            return -scalar
        except Exception as exc:
            # A pathological geometry (degenerate XML) should not kill the run
            logger.warning("eval_failed", x=np.round(x, 3).tolist(), error=str(exc))
            return 0.0  # worst possible coverage

