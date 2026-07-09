"""Coordinate-frame conventions and transforms for the neuro-manipulator project.

Two frames appear throughout the pipeline:
1. Lab Frame (BTS SMART-DX EVO optoelectronic capture):
    +X = lateral (subject's left)
    +Y = vertical (up)
    +Z = posterior (behind the subject)

    ADL kinematics are captured here and stored with the sternum marker as the reference, so the in-memory Delaunay
    tessellations live in this frame.

2. MuJoCo / ada_assets frame:
    +X = forward
    +Y = lateral
    +Z = up

`LAB_TO_MUJOCO` is the single source of truth for the rotation between them. It is a prper rotation (det == +1), so its
    inverse is its transpose. Prefer the named helpers `lab_to_mujoco` and `mujoco_to_lab` instead of using the matrix
    directly.

This module sits at the bottom of the dependency graph so every other package can rely on it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

LAB_TO_MUJOCO: npt.NDArray[np.float64] = np.array(
    [
        [0, 0, -1], # MuJoCo X (forward) = -Lab Z
        [-1, 0, 0], # MuJoCo Y (lateral) = -Lab X
        [0, 1, 0],  # MuJoCo Z (up)      = +Lab Y
    ],
dtype=np.float64
)
"""Rotation mapping Lab-frame column vectors to MuJoCo-frame column vectors."""

def lab_to_mujoco(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map (N, 3) points from the lab frame into the MuJoCo frame.

    Args:
        points: Array whose last axis is a lab-frame coordinate triple
    Returns:
        Array of the same shape expressed in the MuJoCo frame.
    """
    return np.asarray(points, dtype=np.float64) @ LAB_TO_MUJOCO.T # row-major notation

def mujoco_to_lab(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map (N, 3) points from the MuJoCo frame into the lab frame.

    The inverse rotation is just the transpose of LAB_TO_MUJOCO.

    Args:
        points: Array whose last axis is a MuJoCo-frame coordinate triple
    Returns:
        Array of the same shape expressed in the lab frame.
    """
    return np.asarray(points, dtype=np.float64) @ LAB_TO_MUJOCO

def validate_transform(*, atol: float = 1e-6) -> None:
    """Assert that LAB_TO_MUJOCO is a proper rotation matrix with a clean roundtrip.

    Call once at startup for any pipeline that bridges frames. Guards against two common pitfalls:
        1. A reflection sneaking in: (det == -1).
        2. Inverse being applied in the wrong direction.

    Raises:
        ValueError: If the determinant is not +1 or the roundtrip does not return the original point within `atol`.

    """
    det = float(np.linalg.det(LAB_TO_MUJOCO))
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(f"LAB_TO_MUJOCO is not a proper rotation matrix (det={det}), expected +1.")

    probe = np.array([0.10, -0.20, 0.30]) # arbitrary point in the lab frame
    if not np.allclose(mujoco_to_lab(lab_to_mujoco(probe)), probe, atol=atol):
        raise ValueError(f"LAB_TO_MUJOCO roundtrip failed. Wrong inverse")

