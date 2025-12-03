from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from volume.volume_helper import Slice
from index.search import SearchResult


@dataclass(frozen=True)
class SpatialError:
    """
    Simple spatial error in voxel space, based on 3D centers.

    We treat each slice/patch as a 3D point:
        p_query = q_slice.center_xyz_vox
        p_retr  = (center_x_vox, center_y_vox, center_z_vox) from metadata

    Fields
    ------
    dist : Euclidean distance ||p_query - p_retr||
    dx,dy,dz : component-wise differences in voxels.
    """
    dist: float
    dx: float
    dy: float
    dz: float


def compute_spatial_error(
    q_slice: Slice,
    hit: SearchResult,
) -> SpatialError:
    """
    Spatial error between query slice and retrieved patch, using 3D
    centers in voxel space.

    - Query center:
        q_slice.center_xyz_vox  (tuple of 3 floats)
    - Retrieved center:
        center_x_vox, center_y_vox, center_z_vox from hit.meta

    Returns Euclidean distance in voxels and its components.
    """
    # --- Query 3D center ---
    try:
        q_pos = np.asarray(q_slice.center_xyz_vox, dtype=np.float64)
        if q_pos.shape != (3,):
            raise ValueError("q_slice.center_xyz_vox must be a length-3 tuple.")
    except Exception:
        nan = float("nan")
        return SpatialError(dist=nan, dx=nan, dy=nan, dz=nan)

    # --- Retrieved 3D center from metadata ---
    m = hit.meta
    try:
        r_pos = np.asarray(
            [
                float(m["center_x_vox"]),
                float(m["center_y_vox"]),
                float(m["center_z_vox"]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        nan = float("nan")
        return SpatialError(dist=nan, dx=nan, dy=nan, dz=nan)

    diff = q_pos - r_pos
    dx, dy, dz = [float(d) for d in diff]
    dist = float(np.linalg.norm(diff))

    return SpatialError(dist=dist, dx=dx, dy=dy, dz=dz)


def compute_region_error(
    labels_q: Optional[np.ndarray],
    labels_r: Optional[np.ndarray],
) -> float:
    """
    Region-composition error between query and retrieved patches, using Allen labels.

    This is an L1-style mismatch fraction in [0,1], i.e. approximate fraction of pixels
    whose region label would have to change to make the two patches match.

    For both query and retrieved patch:
      - Flatten labels.
      - Build histograms of pixel counts per region ID, but only over
        region IDs that appear in at least one of the two patches.
      - Compute:

            total_abs_diff = Σ_r |c_q[r] - c_r[r]|

        where c_q[r], c_r[r] are pixel counts for region r in query
        and retrieved patch, respectively.

      - Normalize by 2 * N, where N is the number of pixels in the image
        (max of the two sizes):

            error = total_abs_diff / (2 * N)

    This yields a value in [0, 1]:

      - 0   -> identical region composition (same counts per region)
      - 0.3 -> about 30% of pixels are "off" in terms of region labels
      - 1   -> maximally different (e.g. all pixels change region)

    Multiplying by 100 gives a percentage of region-pixel mismatch.
    """
    if labels_q is None or labels_r is None:
        return float("nan")

    q_flat = np.asarray(labels_q, dtype=np.int64).ravel()
    r_flat = np.asarray(labels_r, dtype=np.int64).ravel()

    # Map negative labels to 0 just in case.
    q_flat = np.where(q_flat < 0, 0, q_flat)
    r_flat = np.where(r_flat < 0, 0, r_flat)

    N_q = int(q_flat.size)
    N_r = int(r_flat.size)

    if N_q == 0 and N_r == 0:
        # Both empty -> no error
        return 0.0

    N = max(N_q, N_r, 1)

    q_vals, q_counts = np.unique(q_flat, return_counts=True)
    r_vals, r_counts = np.unique(r_flat, return_counts=True)

    q_hist = {int(v): int(c) for v, c in zip(q_vals, q_counts)}
    r_hist = {int(v): int(c) for v, c in zip(r_vals, r_counts)}

    all_regions = q_hist.keys() | r_hist.keys()
    if not all_regions:
        return 0.0

    total_abs_diff = 0
    for rid in all_regions:
        cq = q_hist.get(rid, 0)
        cr = r_hist.get(rid, 0)
        total_abs_diff += abs(cq - cr)

    error = total_abs_diff / (2.0 * float(N))
    # Numerical safety
    return float(np.clip(error, 0.0, 1.0))
