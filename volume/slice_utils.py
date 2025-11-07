# volume/slice_utils.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np

# ---------- private helpers (intentionally minimal) ----------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def _box_depth_bounds(vol_shape_zyx: Tuple[int, int, int], n_xyz: np.ndarray) -> tuple[float, float]:
    """
    Project the volume's 8 centered corners onto n -> min/max depth (voxels).
    """
    Z, Y, X = vol_shape_zyx
    cx, cy, cz = (X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0
    corners = np.array(
        [[dx, dy, dz] for dx in (-cx, cx) for dy in (-cy, cy) for dz in (-cz, cz)],
        dtype=np.float64,
    )
    n = _unit(n_xyz.astype(np.float64))
    d = corners @ n
    return float(d.min()), float(d.max())

# ---------- public api ----------

def normals_spherical_fibonacci(k: int) -> List[tuple[float, float, float]]:
    """
    ~Uniform directions on the sphere (S^2).
    Returns a list of (x, y, z) unit normals.
    """
    i = np.arange(k, dtype=np.float64) + 0.5
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(1.0 - z * z)
    x, y = r * np.cos(theta), r * np.sin(theta)
    V = np.stack([x, y, z], axis=1)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return [tuple(map(float, row)) for row in V]

def normals_orthogonal(include_diagonals: bool = False) -> List[tuple[float, float, float]]:
    """
    (X, Y, Z) axes; optional face/body diagonals.
    """
    base = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    if not include_diagonals:
        return base
    diag2 = [
        ( 1,  1,  0), ( 1, -1,  0), (-1,  1,  0), (-1, -1,  0),
        ( 1,  0,  1), ( 1,  0, -1), (-1,  0,  1), (-1,  0, -1),
        ( 0,  1,  1), ( 0,  1, -1), ( 0, -1,  1), ( 0, -1, -1),
    ]
    diag3 = [
        ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
        (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
    ]
    def u(t): 
        v = np.asarray(t, float); v /= (np.linalg.norm(v) + 1e-12); return tuple(v.tolist())
    return base + [u(t) for t in (diag2 + diag3)]

def depths_for_normal(
    vol_shape_zyx: Tuple[int, int, int],
    normal_xyz: Tuple[float, float, float],
    *,
    num: int | None = None,     # choose exactly one of (num, step)
    step: float | None = None,
    margin: float = 0.0,
) -> np.ndarray:
    """
    Evenly spaced depths along 'normal' that remain inside the volume box.
    - num: fixed count in [dmin+margin, dmax-margin]
    - step: fixed spacing from dmin+margin upward
    """
    n = np.asarray(normal_xyz, dtype=np.float64)
    dmin, dmax = _box_depth_bounds(vol_shape_zyx, n)
    dmin += margin; dmax -= margin
    if num is not None:
        return np.linspace(dmin, dmax, int(num), dtype=np.float64)
    count = int(np.floor((dmax - dmin) / float(step))) + 1
    return dmin + np.arange(count, dtype=np.float64) * float(step)

def iter_slices(
    vol_helper,                                   # VolumeHelper
    normals_xyz: Iterable[tuple[float, float, float]],
    *,
    num: int | None = None,                       # choose one of (num, step)
    step: float | None = None,
    size_px: int = 512,
    pixel: float = 1.0,
    include_annotation: bool = False,
    annotation_helper=None,
):
    """
    Rotation-free generator of Slice objects.
    """
    Z, Y, X = vol_helper.get_dimension()
    for n in normals_xyz:
        for d in depths_for_normal((Z, Y, X), n, num=num, step=step):
            yield vol_helper.get_slice(
                normal=n,
                depth=float(d),
                rotation=0.0,
                size=int(size_px),
                pixel=float(pixel),
                linear_interp=True,
                include_annotation=include_annotation,
                annotation_helper=annotation_helper,
            )
