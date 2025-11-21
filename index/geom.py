# index/geom.py
from __future__ import annotations
from typing import Generator, Optional, Tuple, List, Dict

import math
import numpy as np

from volume.volume_helper import VolumeHelper, AnnotationHelper, Slice
from index.config import FIXED_ROTATIONS, FIXED_PIXEL_STEP_VOX, FIXED_STEP_VOX, FIXED_MARGIN_VOX

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
    step_vox: float = 1.0,
    margin_vox: float = 0.0,
) -> np.ndarray:
    """
    Step-based depths (signed, voxels along n) guaranteed to remain inside the box.
    Uses a fixed step size and margin.
    """
    if step_vox <= 0:
        raise ValueError("step_vox must be > 0")
    n = np.asarray(normal_xyz_unit, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    dmin, dmax = _box_depth_bounds_for_normal(vol_shape_zyx, n)
    dmin += margin_vox
    dmax -= margin_vox
    count = int(math.floor((dmax - dmin) / step_vox)) + 1
    return dmin + np.arange(count, dtype=np.float64) * step_vox


# ------------------------- planning helper -------------------------


def plan_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Precompute geometry for Fibonacci-sphere slicing.

    Returns
    -------
    (plan, total_slices)
      - plan: list of (normal_xyz_unit, depths_1d) where:
          normal_xyz_unit: np.ndarray shape (3,)
          depths_1d       : np.ndarray of depths along that normal
      - total_slices: total number of (normal, depth, rotation) combinations
                      using FIXED_ROTATIONS.
    """
    Z, Y, X = vol_helper.get_dimension()
    normals = _spherical_fibonacci_normals(k_normals)

    plan: List[Tuple[np.ndarray, np.ndarray]] = []
    total_slices = 0

    for n in normals:
        n_arr = np.asarray(n, dtype=np.float64)
        n_arr /= (np.linalg.norm(n_arr) + 1e-12)

        depths = _depth_schedule_step(
            (Z, Y, X),
            tuple(n_arr.tolist()),
            step_vox=FIXED_STEP_VOX,
            margin_vox=FIXED_MARGIN_VOX,
        )
        plan.append((n_arr, depths))
        total_slices += len(depths) * len(FIXED_ROTATIONS)

    return plan, total_slices


# ------------------------- public API -------------------------


def count_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
) -> int:
    """
    Exact count of slices iter_slices_fibonacci(...) will yield,
    using the same fixed internal settings.
    """
    _, total = plan_slices_fibonacci(vol_helper, k_normals=k_normals)
    return total


def iter_slices_fibonacci(
    vol_helper: VolumeHelper,
    *,
    k_normals: int,
    size_px: int = 512,
    linear_interp: bool = True,
    include_annotation: bool = False,
    annotation_helper: Optional[AnnotationHelper] = None,
) -> Tuple[Generator[Tuple[Slice, Dict[str, float]], None, None], int]:
    """
    Return (iterator, total_slices) for Fibonacci-sphere sampling.

    Fixed internal settings:
      - depths: every voxel plane along each normal (step_vox=1.0, margin=0.0)
      - rotations: FIXED_ROTATIONS
      - pixel spacing: 1.0 voxel/pixel

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

    Returns
    -------
    (it, total_slices)
        it           : generator yielding (Slice, info_dict)
        total_slices : exact number of slices that will be yielded
    """
    plan, total_slices = plan_slices_fibonacci(vol_helper, k_normals=k_normals)

    def _gen() -> Generator[Tuple[Slice, Dict[str, float]], None, None]:
        for ni, (n_arr, depths) in enumerate(plan):
            n_arr = n_arr.astype(np.float64)
            n_arr /= (np.linalg.norm(n_arr) + 1e-12)
            normal_tuple = tuple(float(x) for x in n_arr.tolist())

            for di, d in enumerate(depths):
                for ri, rot in enumerate(FIXED_ROTATIONS):
                    s = vol_helper.get_slice(
                        normal=normal_tuple,
                        depth=float(d),
                        rotation=float(rot),
                        size=int(size_px),
                        pixel=FIXED_PIXEL_STEP_VOX,
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
