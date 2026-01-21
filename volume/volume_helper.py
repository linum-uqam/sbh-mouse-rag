# volume/volume_helper.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Iterable
from numpy.typing import NDArray

import math
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from allensdk.core.reference_space_cache import ReferenceSpaceCache  # AllenVolume + Annotation
import nibabel as nib  # NiftiVolume
import matplotlib.pyplot as plt


# -------------------- misc --------------------

def _compute_lo_hi(arr: np.ndarray, percentiles=(0.5, 99.5)) -> tuple[float, float]:
    lo, hi = np.percentile(arr, percentiles)
    return float(lo), float(hi)


def _safe_float(x: float, eps: float = 1e-12) -> float:
    return float(x) if abs(float(x)) > eps else float(eps)


def _trimmed_mean(d: np.ndarray, trim: float) -> float:
    """
    Mean of the smallest (1-trim) fraction. Equivalent to dropping the largest trim fraction.
    trim in [0, 0.5) is typical.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    if trim <= 0.0:
        return float(d.mean())
    trim = float(np.clip(trim, 0.0, 0.49))
    k = int(max(1, math.floor(d.size * (1.0 - trim))))
    # take k smallest distances
    part = np.partition(d, k - 1)[:k]
    return float(part.mean())


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / (s + 1e-12)


def adaptive_temperature_from_distances(
    distances: Iterable[float],
    *,
    target_ratio_median_over_best: float = 10.0,
    min_tau: float = 1e-6,
) -> float:
    """
    Query-adaptive temperature.

    Chooses tau so that:
      p(best) / p(median) ~= target_ratio_median_over_best

    With p_i ∝ exp(-d_i / tau), we get:
      exp((d_median - d_min)/tau) = target_ratio
      => tau = (d_median - d_min) / ln(target_ratio)

    This tends to produce a stable, not-too-peaky distribution over top-100.
    """
    d = np.asarray(list(distances), dtype=np.float64)
    if d.size == 0:
        return float(min_tau)
    d_min = float(np.min(d))
    d_med = float(np.median(d))
    denom = math.log(_safe_float(target_ratio_median_over_best))
    tau = (d_med - d_min) / (denom + 1e-12)
    return float(max(tau, min_tau))


def distances_to_distribution(
    distances: Iterable[float],
    *,
    tau: Optional[float] = None,
    target_ratio_median_over_best: float = 10.0,
) -> np.ndarray:
    """
    Convert distances -> probability distribution p over candidates.
    p_i = softmax(-d_i / tau)

    If tau is None, uses adaptive_temperature_from_distances().
    """
    d = np.asarray(list(distances), dtype=np.float64)
    if d.size == 0:
        return d
    if tau is None:
        tau = adaptive_temperature_from_distances(
            d, target_ratio_median_over_best=target_ratio_median_over_best
        )
    logits = -d / float(max(tau, 1e-12))
    return _softmax(logits)


# -------------------- geometry --------------------

def slice_pixel_to_voxel(
    volume_shape_zyx: Tuple[int, int, int],
    normal_xyz_unit: Tuple[float, float, float],
    depth_vox: float,
    rotation_deg: float,
    plane_size_px: int,
    pixel_step_vox: float,
    x_px: float,
    y_px: float,
) -> np.ndarray:
    """
    Map a pixel (x_px, y_px) in a slice plane to a 3D voxel coordinate (X,Y,Z).

    The slice plane is defined by:
      - normal_xyz_unit: unit normal (X,Y,Z)
      - depth_vox      : signed distance along normal from volume center
      - rotation_deg   : in-plane rotation around normal
      - plane_size_px  : side length (in pixels) of the sampling plane
      - pixel_step_vox : voxel step per pixel in the plane

    volume_shape_zyx is (Z, Y, X).
    """
    Z, Y, X = volume_shape_zyx

    n = np.asarray(normal_xyz_unit, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    # build orthonormal basis {u, v, n} in voxel (X,Y,Z) space
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)

    # in-plane rotation
    th = math.radians(rotation_deg)
    u_r = math.cos(th) * u + math.sin(th) * v
    v_r = -math.sin(th) * u + math.cos(th) * v

    # center of volume in XYZ
    center_xyz = np.array(
        [(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0],
        dtype=np.float64,
    )

    # offsets along u_r, v_r from slice center
    half = (plane_size_px - 1) / 2.0
    su = (x_px - half) * float(pixel_step_vox)
    sv = (y_px - half) * float(pixel_step_vox)

    # final point in XYZ
    P_xyz = center_xyz + depth_vox * n + su * u_r + sv * v_r
    return P_xyz  # (X, Y, Z)


# -------------------- Slice object --------------------

@dataclass(frozen=True, slots=True)
class Slice:
    image: NDArray[np.float32]                         # (H,W) float32 in [0,1] for grayscale
    normal_xyz_unit: Tuple[float, float, float]
    depth_vox: float
    rotation_deg: float
    pixel_step_vox: float

    # size_px is the sampling plane size (original plane), not the crop size.
    size_px: int

    volume_shape_zyx: Tuple[int, int, int]
    spacing_zyx: Tuple[float, float, float]            # (Z,Y,X) spacing (mm or um; consistent within your pipeline)

    # crop position within the original plane (x0,y0) in plane pixel coordinates
    origin_px_in_plane: Tuple[int, int] = (0, 0)

    # 3D center of this slice/crop in voxel coordinates (X, Y, Z)
    center_xyz_vox: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    labels: Optional[np.ndarray] = None       # (H,W) int32, 0=background; None if absent

    # ---------- coordinate helpers ----------
    def _spacing_xyz(self) -> Tuple[float, float, float]:
        # spacing_zyx = (sz, sy, sx)
        sz, sy, sx = self.spacing_zyx
        return (float(sx), float(sy), float(sz))

    def _unit(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    def _basis_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (n, u_r, v_r, center_xyz) in voxel XYZ coordinates.
        """
        Z, Y, X = self.volume_shape_zyx
        n = self._unit(np.asarray(self.normal_xyz_unit, dtype=np.float64))

        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = self._unit(ref - np.dot(ref, n) * n)
        v = np.cross(n, u)

        th = math.radians(float(self.rotation_deg))
        u_r = math.cos(th) * u + math.sin(th) * v
        v_r = -math.sin(th) * u + math.cos(th) * v

        center_xyz = np.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], dtype=np.float64)
        return n, u_r, v_r, center_xyz

    def pixel_to_voxel(self, x_px: float, y_px: float) -> Tuple[float, float, float]:
        """
        Map a point in this slice image pixel coordinates (x_px, y_px) to 3D voxel XYZ.

        For crops, this correctly accounts for origin_px_in_plane.
        """
        x0, y0 = self.origin_px_in_plane
        plane_x = float(x0) + float(x_px)
        plane_y = float(y0) + float(y_px)
        P_xyz = slice_pixel_to_voxel(
            volume_shape_zyx=self.volume_shape_zyx,
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=float(self.depth_vox),
            rotation_deg=float(self.rotation_deg),
            plane_size_px=int(self.size_px),
            pixel_step_vox=float(self.pixel_step_vox),
            x_px=float(plane_x),
            y_px=float(plane_y),
        )
        return (float(P_xyz[0]), float(P_xyz[1]), float(P_xyz[2]))

    def sample_points_xyz(
        self,
        grid: int = 64,
        *,
        physical: bool = True,
    ) -> np.ndarray:
        """
        Sample a grid x grid set of 3D points on the slice support, using the *current* slice image extent.

        Returns: (grid*grid, 3) array in XYZ.
          - If physical=True, units are physical (spacing applied).
          - Else, units are voxel indices.
        """
        H, W = self.image.shape
        g = int(grid)
        if g <= 1:
            raise ValueError("grid must be >= 2")

        xs = np.linspace(0.0, float(W - 1), g, dtype=np.float64)
        ys = np.linspace(0.0, float(H - 1), g, dtype=np.float64)
        XX, YY = np.meshgrid(xs, ys)  # local slice pixels

        x0, y0 = self.origin_px_in_plane
        plane_x = XX + float(x0)
        plane_y = YY + float(y0)

        n, u_r, v_r, center_xyz = self._basis_xyz()
        half = (float(self.size_px) - 1.0) / 2.0
        su = (plane_x - half) * float(self.pixel_step_vox)
        sv = (plane_y - half) * float(self.pixel_step_vox)

        P = center_xyz + float(self.depth_vox) * n + su[..., None] * u_r + sv[..., None] * v_r  # (H,W,3)
        pts = P.reshape(-1, 3)

        if physical:
            sx, sy, sz = self._spacing_xyz()
            pts = pts * np.array([sx, sy, sz], dtype=np.float64)
        return pts

    # ---------- I/O ----------
    def save(
        self,
        path: str | Path,
        title: str | None = None,
        dpi: int = 200,
        overlay: str = "image",   # "image" | "labels" | "image+labels"
        alpha: float = 0.5,
    ) -> Path:
        """
        Save slice image/labels.
        overlay:
          - "image"         -> grayscale image
          - "labels"        -> colored labels only
          - "image+labels"  -> labels over grayscale
        """
        path = Path(path)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if overlay == "image":
            ax.imshow(self.image, cmap="gray")
        elif overlay == "labels":
            if self.labels is None:
                raise ValueError("No labels in this slice to save as 'labels'.")
            rgba = self._labels_to_rgba(self.labels, alpha=1.0)  # opaque
            ax.imshow(rgba)
        elif overlay == "image+labels":
            if self.labels is None:
                raise ValueError("No labels in this slice to overlay.")
            rgba = self._labels_to_rgba(self.labels, alpha=alpha)
            rgb = self._overlay_rgba(self.image, rgba)
            ax.imshow(rgb)
        else:
            raise ValueError("overlay must be one of {'image','labels','image+labels'}")

        if title:
            ax.set_title(title)
        ax.axis("off")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path

    # ---------- Deterministic crop ----------
    def crop_norm(
        self,
        cx: float, cy: float,   # center in [0,1] (x then y)
        rw: float, rh: float,   # relative width/height in (0,1]
        *,
        clamp: bool = True,
    ) -> "Slice":
        """
        Deterministic crop using normalized args; crops image and labels (if present).

        Stores crop offset (origin_px_in_plane) so spatial methods remain correct.
        """
        H, W = self.image.shape
        w = max(1, int(round(W * rw)))
        h = max(1, int(round(H * rh)))

        # crop center in *current slice image* coordinates
        x_center = int(round(cx * (W - 1)))
        y_center = int(round(cy * (H - 1)))
        x0 = x_center - w // 2
        y0 = y_center - h // 2
        x1 = x0 + w
        y1 = y0 + h

        if clamp:
            x0 = max(0, min(x0, W))
            y0 = max(0, min(y0, H))
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            if x1 <= x0 or y1 <= y0:
                x0, y0 = min(W - 1, x0), min(H - 1, y0)
                x1, y1 = min(W, x0 + 1), min(H, y0 + 1)
        else:
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                raise ValueError("Crop window is out of bounds and clamp=False.")

        img_sub = self.image[y0:y1, x0:x1]
        labels_sub = None
        if self.labels is not None:
            labels_sub = self.labels[y0:y1, x0:x1].copy()

        # origin accumulates (crop of crop works)
        ox, oy = self.origin_px_in_plane
        new_origin = (int(ox + x0), int(oy + y0))

        # 3D center is the 3D position of the crop center (in plane coordinates)
        new_H, new_W = img_sub.shape
        local_cx = (new_W - 1) / 2.0
        local_cy = (new_H - 1) / 2.0
        plane_cx = float(new_origin[0]) + float(local_cx)
        plane_cy = float(new_origin[1]) + float(local_cy)

        center_xyz = slice_pixel_to_voxel(
            volume_shape_zyx=self.volume_shape_zyx,
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            plane_size_px=self.size_px,             # plane size = original sampling size
            pixel_step_vox=self.pixel_step_vox,
            x_px=float(plane_cx),
            y_px=float(plane_cy),
        )

        return Slice(
            image=img_sub.copy(),
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            pixel_step_vox=self.pixel_step_vox,
            size_px=self.size_px,                   # keep plane size
            volume_shape_zyx=self.volume_shape_zyx,
            spacing_zyx=self.spacing_zyx,
            origin_px_in_plane=new_origin,
            center_xyz_vox=(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])),
            labels=labels_sub,
        )

    def normalized(
        self,
        *,
        percentiles: tuple[float, float] = (0.5, 99.5),
    ) -> "Slice":
        """
        Return a NEW Slice with image normalized to [0,1].
        - Always clips to [0,1].
        """
        img = self.image.astype(np.float32, copy=False)
        lo, hi = np.percentile(img, percentiles)
        scale = max(float(hi - lo), 1e-12)
        img01 = np.clip((img - float(lo)) / scale, 0.0, 1.0).astype(np.float32)

        return Slice(
            image=img01,
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            pixel_step_vox=self.pixel_step_vox,
            size_px=self.size_px,
            volume_shape_zyx=self.volume_shape_zyx,
            spacing_zyx=self.spacing_zyx,
            origin_px_in_plane=self.origin_px_in_plane,
            center_xyz_vox=self.center_xyz_vox,
            labels=(None if self.labels is None else self.labels.copy()),
        )

    # ---------- Distance ----------
    @staticmethod
    def _mirror_points_axis2_x(points_xyz: np.ndarray, *, volume_shape_zyx: Tuple[int, int, int], spacing_zyx: Tuple[float, float, float], physical: bool) -> np.ndarray:
        """
        Mirror points along axis=2 of (Z,Y,X), i.e. X axis in XYZ.
        Reflection plane is x = (X-1)/2.
        If physical=True, mirror in physical coordinates using sx.
        """
        pts = np.asarray(points_xyz, dtype=np.float64).copy()
        X = int(volume_shape_zyx[2])
        if physical:
            sx = float(spacing_zyx[2])  # spacing_zyx = (sz,sy,sx)
            x_max = float((X - 1) * sx)
            pts[:, 0] = x_max - pts[:, 0]
        else:
            x_max = float(X - 1)
            pts[:, 0] = x_max - pts[:, 0]
        return pts

    @staticmethod
    def distance(
        a: "Slice",
        b: "Slice",
        *,
        grid: int = 64,
        trim: float = 0.10,
        physical: bool = True,
        also_return_mirror_diagnostic: bool = False,
    ) -> float | Tuple[float, Dict[str, float]]:
        """
        Robust bidirectional Chamfer distance between two slices, computed from sampled 3D points.

        - Samples a grid×grid point set on each slice support (respecting crops).
        - Uses 1-NN distances via KD-trees (fast).
        - Returns a robust score using trimmed mean to reduce crop-boundary sensitivity.

        If you want strict laterality supervision: use the returned 'distance' as ground-truth.
        If also_return_mirror_diagnostic=True, also returns 'distance_mirror_b' where b is mirrored along X
        (axis=2 in ZYX). This is ONLY a diagnostic; do not use min() of the two if laterality matters.
        """
        Qa = a.sample_points_xyz(grid=grid, physical=physical)
        Cb = b.sample_points_xyz(grid=grid, physical=physical)

        tree_b = cKDTree(Cb)
        d_a_to_b, _ = tree_b.query(Qa, k=1, workers=-1)

        tree_a = cKDTree(Qa)
        d_b_to_a, _ = tree_a.query(Cb, k=1, workers=-1)

        dist = _trimmed_mean(d_a_to_b, trim) + _trimmed_mean(d_b_to_a, trim)

        if not also_return_mirror_diagnostic:
            return float(dist)

        # mirror diagnostic (mirror b across X)
        Cb_m = Slice._mirror_points_axis2_x(
            Cb, volume_shape_zyx=b.volume_shape_zyx, spacing_zyx=b.spacing_zyx, physical=physical
        )
        tree_bm = cKDTree(Cb_m)
        d_a_to_bm, _ = tree_bm.query(Qa, k=1, workers=-1)

        tree_a2 = cKDTree(Qa)
        d_bm_to_a, _ = tree_a2.query(Cb_m, k=1, workers=-1)
        dist_m = _trimmed_mean(d_a_to_bm, trim) + _trimmed_mean(d_bm_to_a, trim)

        info = {
            "distance": float(dist),
            "distance_mirror_b": float(dist_m),
        }
        return float(dist), info

    @staticmethod
    def distance_pose(a: "Slice", b: "Slice") -> float:
        """
        Old approximate metric (kept for debugging/ablation).
        """
        n1 = np.asarray(a.normal_xyz_unit, float)
        n2 = np.asarray(b.normal_xyz_unit, float)
        d1 = float(a.depth_vox)
        d2 = float(b.depth_vox)
        pos = float(np.linalg.norm(d1 * n1 - d2 * n2))
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        theta = math.acos(dot)
        Za, Ya, Xa = a.volume_shape_zyx
        Zb, Yb, Xb = b.volume_shape_zyx
        k = 0.5 * max(max(Xa, Ya, Za), max(Xb, Yb, Zb))
        return pos + k * theta

    # ---------- internal: label color + overlay ----------
    @staticmethod
    def _labels_to_rgba(labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        H, W = labels.shape
        lab = labels.astype(np.int64)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        uniques = np.unique(lab)
        uniques = uniques[uniques != 0]
        if uniques.size == 0:
            return rgba  # all transparent

        def color_from_id(k: int) -> Tuple[int, int, int]:
            h = (1103515245 * (k ^ 0x9e3779b97f4a7c15) + 12345) & 0xFFFFFFFF
            hue = (h % 360) / 360.0
            sat = 0.65 + ((h >> 8) % 35) / 100.0
            val = 0.65 + ((h >> 16) % 35) / 100.0
            i = int(hue * 6)
            f = hue * 6 - i
            p = val * (1 - sat)
            q = val * (1 - f * sat)
            t = val * (1 - (1 - f) * sat)
            i = i % 6
            if i == 0:
                r, g, b = val, t, p
            elif i == 1:
                r, g, b = q, val, p
            elif i == 2:
                r, g, b = p, val, t
            elif i == 3:
                r, g, b = p, q, val
            elif i == 4:
                r, g, b = t, p, val
            else:
                r, g, b = val, p, q
            return int(r * 255), int(g * 255), int(b * 255)

        lut = {int(k): color_from_id(int(k)) for k in uniques}
        a_val = int(np.clip(alpha, 0, 1) * 255)
        for k, (r, g, b) in lut.items():
            mask = (lab == k)
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = a_val
        return rgba

    @staticmethod
    def _overlay_rgba(gray01: np.ndarray, rgba: np.ndarray) -> np.ndarray:
        base = np.clip(gray01, 0, 1)
        H, W = base.shape
        out = np.zeros((H, W, 3), dtype=np.float32)
        out[..., 0] = base
        out[..., 1] = base
        out[..., 2] = base
        a = (rgba[..., 3:4].astype(np.float32) / 255.0)
        out = (1 - a) * out + a * (rgba[..., :3].astype(np.float32) / 255.0)
        return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# -------------------- Core helpers --------------------

@dataclass
class _Vol:
    arr: np.ndarray        # (Z, Y, X) float32/int32
    spacing: Tuple[float, float, float]  # (Z, Y, X) voxel spacing (optional info)


class VolumeHelper:
    """
    Base class: provides slicing + cropping.
    Subclasses must call _set_volume(...) after loading.
    """

    def __init__(self):
        self._vol: Optional[_Vol] = None
        self._global_lo: Optional[float] = None
        self._global_hi: Optional[float] = None
        self._is_normalized: bool = False

    def _set_volume(self, arr_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        if arr_zyx.ndim != 3:
            raise ValueError("Volume must be 3D")
        self._vol = _Vol(arr=arr_zyx, spacing=tuple(map(float, spacing_zyx)))
        lo, hi = _compute_lo_hi(arr_zyx, percentiles=(0.5, 99.5))
        self._global_lo, self._global_hi = float(lo), float(hi)
        self._is_normalized = False

    def is_valid_slice(
        self,
        sl: Slice,
        *,
        ratio_threshold: float = 0.25,
        value_threshold_pct: float = 0.10,
    ) -> bool:
        """
        Decide whether a slice contains enough tissue vs background.

        Threshold is: thr = lo + value_threshold_pct * (hi - lo).
        """
        if self._global_lo is None or self._global_hi is None:
            raise RuntimeError("Global intensity bounds not initialized")

        lo, hi = self._global_lo, self._global_hi
        thr = lo + value_threshold_pct * (hi - lo)

        img = sl.image
        mask = img > thr
        return (mask.sum() / img.size) > ratio_threshold

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _build_plane_coords(
        self,
        normal_xyz: np.ndarray,
        depth: float,
        rotation_deg: float,
        size: int,
        pixel: float,
    ) -> np.ndarray:
        """
        Build map_coordinates coords for a slice plane.
        Returns coords: (3,H,W) ordered (Z,Y,X).
        """
        if self._vol is None:
            raise RuntimeError("Volume not loaded")
        Z, Y, X = self._vol.arr.shape
        n = self._unit(normal_xyz.astype(np.float64))

        # basis {u, v, n} in voxel space (X,Y,Z)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = self._unit(ref - np.dot(ref, n) * n)
        v = np.cross(n, u)

        th = math.radians(rotation_deg)
        u_r = math.cos(th) * u + math.sin(th) * v
        v_r = -math.sin(th) * u + math.cos(th) * v

        H = W = int(size)
        half = (H - 1) / 2.0
        su = (np.arange(W) - half) * pixel
        sv = (np.arange(H) - half) * pixel
        UU, VV = np.meshgrid(su, sv)

        center_xyz = np.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], dtype=np.float64)
        P = center_xyz + depth * n + UU[..., None] * u_r + VV[..., None] * v_r  # (H,W,3)
        coords = np.stack([P[..., 2], P[..., 1], P[..., 0]], axis=0)  # (3,H,W) = (Z,Y,X)
        return coords

    # --------------- public API ---------------
    def get_dimension(self) -> Tuple[int, int, int]:
        if self._vol is None:
            raise RuntimeError("Volume not loaded")
        return tuple(self._vol.arr.shape)  # (Z,Y,X)

    def normalize_volume(
        self,
        *,
        percentiles: tuple[float, float] = (0.5, 99.5),
    ) -> None:
        if self._vol is None:
            raise RuntimeError("Volume not loaded")
        arr = self._vol.arr.astype(np.float32, copy=False)

        lo, hi = np.percentile(arr, percentiles)
        scale = max(float(hi - lo), 1e-12)

        arr01 = (arr - float(lo)) / scale
        np.clip(arr01, 0.0, 1.0, out=arr01)

        self._vol = _Vol(arr=arr01.astype(np.float32, copy=False), spacing=self._vol.spacing)
        self._global_lo, self._global_hi = 0.0, 1.0
        self._is_normalized = True

    def is_normalized(self) -> bool:
        return bool(self._is_normalized)

    def get_global_intensity_bounds(self) -> Tuple[float, float]:
        if self._global_lo is None or self._global_hi is None:
            raise RuntimeError("Global intensity bounds not initialized")
        return (float(self._global_lo), float(self._global_hi))

    def get_slice(
        self,
        normal: tuple[float, float, float],
        depth: float,
        rotation: float = 0.0,
        size: int = 512,
        pixel: float = 1.0,
        linear_interp: bool = True,
        *,
        include_annotation: bool = False,
        annotation_helper: "AnnotationHelper | None" = None,
    ) -> Slice:
        """
        Sample a slice. If include_annotation=True:
          - AllenVolume will sample its own annotation.
          - NiftiVolume will use the provided annotation_helper (Allen CCF) to overlay labels.
        """
        if self._vol is None:
            raise RuntimeError("Volume not loaded")

        V = self._vol.arr
        Z, Y, X = V.shape
        spacing_zyx = self._vol.spacing

        n = np.asarray(normal, dtype=np.float64)
        n_unit = tuple(self._unit(n).tolist())

        coords = self._build_plane_coords(n, float(depth), float(rotation), int(size), float(pixel))

        order = 1 if (linear_interp and np.issubdtype(V.dtype, np.floating)) else 0
        img = map_coordinates(V, coords, order=order, mode="nearest").astype(np.float32)

        labels = None
        if include_annotation:
            if hasattr(self, "_sample_annotation_slice"):
                labels = getattr(self, "_sample_annotation_slice")(coords)
            elif annotation_helper is not None:
                labels = annotation_helper.sample_labels(coords)

        volume_shape_zyx = (Z, Y, X)
        plane_size_px = int(size)
        pixel_step_vox = float(pixel)

        # center of full slice = middle pixel of the plane
        half = (plane_size_px - 1) / 2.0
        center_xyz = slice_pixel_to_voxel(
            volume_shape_zyx=volume_shape_zyx,
            normal_xyz_unit=n_unit,
            depth_vox=float(depth),
            rotation_deg=float(rotation),
            plane_size_px=plane_size_px,
            pixel_step_vox=pixel_step_vox,
            x_px=float(half),
            y_px=float(half),
        )

        return Slice(
            image=img,
            normal_xyz_unit=n_unit,
            depth_vox=float(depth),
            rotation_deg=float(rotation),
            pixel_step_vox=pixel_step_vox,
            size_px=plane_size_px,
            volume_shape_zyx=volume_shape_zyx,
            spacing_zyx=spacing_zyx,
            origin_px_in_plane=(0, 0),
            center_xyz_vox=(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])),
            labels=labels,
        )


# ------------------------- Allen Annotation Helper -------------------------

class AnnotationHelper:
    """
    Loads Allen CCFv3 annotation volume and provides label sampling.
    Useful to overlay labels on *any* volume slice that is in Allen space.
    """

    def __init__(self, cache_dir: str = "volume/data/allen", resolution_um: int = 25):
        self.cache_dir = Path(cache_dir)
        self.res_um = int(resolution_um)
        self._rc = ReferenceSpaceCache(
            resolution=self.res_um,
            reference_space_key="annotation/ccf_2017",
            manifest=self.cache_dir / "manifest.json",
        )
        self._labels_zyx: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        arr, _ = self._rc.get_annotation_volume()  # (Z,Y,X) int labels
        self._labels_zyx = arr.astype(np.int32, copy=False)

    def sample_labels(self, coords_zyx: np.ndarray) -> np.ndarray:
        """
        coords_zyx: (3,H,W) in (Z,Y,X) index order from VolumeHelper._build_plane_coords/get_slice
        Returns (H,W) int32 label map sampled with nearest neighbor.
        """
        V = self._labels_zyx
        labels = map_coordinates(V, coords_zyx, order=0, mode="nearest").astype(np.int32)
        return labels


# ------------------------- Concrete: Allen -------------------------

class AllenVolume(VolumeHelper):
    """Loads the Allen CCFv3 average template via AllenSDK."""

    def __init__(self, cache_dir: str = "volume/data/allen", resolution_um: int = 25):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.res_um = int(resolution_um)
        self._rc = ReferenceSpaceCache(
            resolution=self.res_um,
            reference_space_key="annotation/ccf_2017",
            manifest=self.cache_dir / "manifest.json",
        )
        self._annot_helper = AnnotationHelper(cache_dir=str(self.cache_dir), resolution_um=self.res_um)
        self.load()

    def load(self):
        arr, _ = self._rc.get_template_volume()   # (Z,Y,X)
        # Allen template is already at the requested resolution; treat spacing as 1 voxel unit unless you want mm/um here.
        self._set_volume(arr.astype(np.float32, copy=False), spacing_zyx=(1.0, 1.0, 1.0))

    def _sample_annotation_slice(self, coords_zyx: np.ndarray) -> np.ndarray:
        return self._annot_helper.sample_labels(coords_zyx)

    def _find_cached(self, filename: str) -> Path:
        matches = list(self.cache_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Could not locate cached file: {filename} under {self.cache_dir}")
        return matches[0]


# ------------------------- Concrete: NIfTI -------------------------

class NiftiVolume(VolumeHelper):
    """Loads a NIfTI (.nii / .nii.gz) volume as (Z,Y,X) float32 (as provided)."""

    def __init__(self, nifti_path: str | Path):
        super().__init__()
        self.path = Path(nifti_path)
        self.load()

    def load(self):
        img = nib.load(str(self.path))
        data_xyz = img.get_fdata(dtype=np.float32)  # nibabel gives (X,Y,Z) in general, but here we do NOT need to transpose.
        # arr_zyx = np.transpose(data_xyz, (2, 1, 0))  # -> (Z,Y,X)
        zooms = img.header.get_zooms()[:3]           # (X,Y,Z) spacing in physical units
        spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        self._set_volume(data_xyz, spacing_zyx=spacing_zyx)
