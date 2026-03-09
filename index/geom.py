# index/geom.py
from __future__ import annotations
from typing import Generator, Optional, Tuple, List, Dict, Sequence

import math
import numpy as np

from volume.volume_helper import VolumeHelper, AnnotationHelper, Slice
from index.config import (
    FIXED_ROTATIONS,
    FIXED_PIXEL_STEP_VOX,
    FIXED_STEP_VOX,
    FIXED_MARGIN_VOX,
)

# ------------------------- internal helpers -------------------------


def _spherical_fibonacci_normals(k: int) -> List[Tuple[float, float, float]]:
    """
    Spherical Fibonacci point set on S^2, mapped to XYZ unit normals.
    Reference: Keinert et al., 'Spherical Fibonacci Mapping'.
    """
    if k <= 0:
        return []
    i = np.arange(k, dtype=np.float64) + 0.5
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    normals = np.stack([x, y, z], axis=1)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    out: List[Tuple[float, float, float]] = []
    seen = set()
    for v in normals:
        key = (round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6))
        if key not in seen:
            seen.add(key)
            out.append((float(v[0]), float(v[1]), float(v[2])))
    return out


def _box_depth_bounds_for_normal(
    vol_shape_zyx: Tuple[int, int, int],
    normal_xyz_unit: np.ndarray,
) -> Tuple[float, float]:
    """
    Given a unit normal in XYZ, return min/max signed depths (voxels) through the
    axis-aligned volume box, measured from the box center along n.
    """
    Z, Y, X = vol_shape_zyx
    cx, cy, cz = ((X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0)
    corners = []
    for dx in (-cx, +cx):
        for dy in (-cy, +cy):
            for dz in (-cz, +cz):
                corners.append(np.array([dx, dy, dz], dtype=np.float64))
    n = normal_xyz_unit.astype(np.float64)
    proj = [float(np.dot(c, n)) for c in corners]
    return min(proj), max(proj)


def _depth_schedule_step(
    vol_shape_zyx: Tuple[int, int, int],
    normal_xyz_unit: Tuple[float, float, float],
    *,
    step_vox: float,
    margin_vox: float,
) -> np.ndarray:
    """
    Step-based depths (signed, voxels along n) guaranteed to remain inside the box.
    Uses a fixed step size and margin.

    Returns a 1D array of depths (voxels) along the normal.
    """
    if step_vox <= 0:
        raise ValueError("step_vox must be > 0")
    if margin_vox < 0:
        raise ValueError("margin_vox must be >= 0")

    n = np.asarray(normal_xyz_unit, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)

    dmin, dmax = _box_depth_bounds_for_normal(vol_shape_zyx, n)
    dmin += margin_vox
    dmax -= margin_vox

    if dmax < dmin:
        return np.array([], dtype=np.float64)

    count = int(math.floor((dmax - dmin) / step_vox)) + 1
    return dmin + np.arange(count, dtype=np.float64) * step_vox


# ------------------------- planning helper -------------------------


def plan_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
    step_vox: float = FIXED_STEP_VOX,
    margin_vox: float = FIXED_MARGIN_VOX,
    rotations_deg: Sequence[float] = FIXED_ROTATIONS,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Precompute geometry for Fibonacci-sphere slicing.

    Parameters
    ----------
    vol_helper : VolumeHelper
        Loaded volume.
    k_normals : int
        Number of (approximately uniform) directions on the sphere.
    step_vox : float
        Step size between consecutive depths, in voxels (signed along the normal).
    margin_vox : float
        Margin removed from both ends of the valid depth interval (in voxels).
    rotations_deg : Sequence[float]
        In-plane rotations (degrees) to apply for each (normal, depth).

    Returns
    -------
    (plan, total_slices)
      - plan: list of (normal_xyz_unit, depths_1d)
      - total_slices: total number of (normal, depth, rotation) combinations
    """
    Z, Y, X = vol_helper.get_dimension()
    normals = _spherical_fibonacci_normals(k_normals)

    rots = list(rotations_deg)
    if len(rots) == 0:
        # keep pipeline valid: at least one rotation
        rots = [0.0]

    plan: List[Tuple[np.ndarray, np.ndarray]] = []
    total_slices = 0

    for n in normals:
        n_arr = np.asarray(n, dtype=np.float64)
        n_arr /= (np.linalg.norm(n_arr) + 1e-12)

        depths = _depth_schedule_step(
            (Z, Y, X),
            tuple(n_arr.tolist()),
            step_vox=float(step_vox),
            margin_vox=float(margin_vox),
        )

        plan.append((n_arr, depths))
        total_slices += int(len(depths)) * int(len(rots))

    return plan, total_slices


# ------------------------- public API -------------------------


def count_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
    step_vox: float = FIXED_STEP_VOX,
    margin_vox: float = FIXED_MARGIN_VOX,
    rotations_deg: Sequence[float] = FIXED_ROTATIONS,
) -> int:
    """
    Exact count of slices iter_slices_fibonacci(...) will yield,
    using the same settings.
    """
    _, total = plan_slices_fibonacci(
        vol_helper,
        k_normals=k_normals,
        step_vox=step_vox,
        margin_vox=margin_vox,
        rotations_deg=rotations_deg,
    )
    return total


def iter_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
    size_px: int = 512,
    linear_interp: bool = True,
    include_annotation: bool = False,
    annotation_helper: Optional[AnnotationHelper] = None,
    step_vox: float = FIXED_STEP_VOX,
    margin_vox: float = FIXED_MARGIN_VOX,
    rotations_deg: Sequence[float] = FIXED_ROTATIONS,
    pixel_step_vox: float = FIXED_PIXEL_STEP_VOX,
) -> Tuple[Generator[Tuple[Slice, Dict[str, float]], None, None], int]:
    """
    Return (iterator, total_slices) for Fibonacci-sphere sampling.

    Parameters
    ----------
    vol_helper : VolumeHelper
        Loaded volume (AllenVolume / NiftiVolume).
    k_normals : int
        Number of (approximately uniform) directions on the sphere.
    size_px : int
        Output slice size (square).
    linear_interp : bool
        Use linear interpolation for float volumes (nearest otherwise).
    include_annotation : bool
        If True, sample labels (Allen CCF) when available.
    annotation_helper : Optional[AnnotationHelper]
        Required for Nifti volumes if include_annotation=True.
    step_vox : float
        Step size between consecutive depths, in voxels.
    margin_vox : float
        Margin removed from valid depth interval (voxels).
    rotations_deg : Sequence[float]
        In-plane rotations (degrees) applied at each (normal, depth).
    pixel_step_vox : float
        Pixel spacing in voxel units (voxel per pixel).

    Returns
    -------
    (it, total_slices)
        it           : generator yielding (Slice, info_dict)
        total_slices : exact number of slices that will be yielded
    """
    plan, total_slices = plan_slices_fibonacci(
        vol_helper,
        k_normals=k_normals,
        step_vox=step_vox,
        margin_vox=margin_vox,
        rotations_deg=rotations_deg,
    )

    rots = list(rotations_deg)
    if len(rots) == 0:
        rots = [0.0]

    def _gen() -> Generator[Tuple[Slice, Dict[str, float]], None, None]:
        for ni, (n_arr, depths) in enumerate(plan):
            n_arr = n_arr.astype(np.float64)
            n_arr /= (np.linalg.norm(n_arr) + 1e-12)
            normal_tuple = tuple(float(x) for x in n_arr.tolist())

            for di, d in enumerate(depths):
                for ri, rot in enumerate(rots):
                    s = vol_helper.get_slice(
                        normal=normal_tuple,
                        depth=float(d),
                        rotation=float(rot),
                        size=int(size_px),
                        pixel=float(pixel_step_vox),
                        linear_interp=linear_interp,
                        include_annotation=include_annotation,
                        annotation_helper=annotation_helper,
                    )
                    info = {
                        "normal_idx": ni,
                        "depth_idx": di,
                        "rot_idx": ri,
                        "normal_xyz_unit": normal_tuple,
                        "depth_vox": float(d),
                        "rotation_deg": float(rot),
                    }
                    yield s, info

    return _gen(), total_slices