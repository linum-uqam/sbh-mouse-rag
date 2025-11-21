# volume/volume_helper.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
from numpy.typing import NDArray
import math
import numpy as np
from scipy.ndimage import map_coordinates
from allensdk.core.reference_space_cache import ReferenceSpaceCache  # AllenVolume + Annotation
import nibabel as nib  # NiftiVolume
import matplotlib.pyplot as plt

def _compute_lo_hi(arr: np.ndarray, percentiles=(0.5, 99.5)) -> tuple[float, float]:
    lo, hi = np.percentile(arr, percentiles)
    return float(lo), float(hi)

# -------------------- Slice object --------------------

@dataclass(frozen=True, slots=True)
class Slice:
    image: NDArray[np.float32]                         # (H,W) float32 in [0,1] for grayscale
    normal_xyz_unit: Tuple[float, float, float]
    depth_vox: float
    rotation_deg: float
    pixel_step_vox: float
    size_px: int
    volume_shape_zyx: Tuple[int, int, int]
    labels: Optional[np.ndarray] = None       # (H,W) int32, 0=background; None if absent

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
        """
        H, W = self.image.shape
        w = max(1, int(round(W * rw)))
        h = max(1, int(round(H * rh)))

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
                x0, y0 = min(W-1, x0), min(H-1, y0)
                x1, y1 = min(W, x0+1), min(H, y0+1)
        else:
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                raise ValueError("Crop window is out of bounds and clamp=False.")

        img_sub = self.image[y0:y1, x0:x1]
        labels_sub = None
        if self.labels is not None:
            labels_sub = self.labels[y0:y1, x0:x1].copy()

        return Slice(
            image=img_sub.copy(),
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            pixel_step_vox=self.pixel_step_vox,
            size_px=min(img_sub.shape[0], img_sub.shape[1]),
            volume_shape_zyx=self.volume_shape_zyx,
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
        scale = max(hi - lo, 1e-12)                     # guard zero/neg span
        img01 = np.clip((img - lo) / scale, 0.0, 1.0).astype(np.float32)

        return Slice(
            image=img01,
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            pixel_step_vox=self.pixel_step_vox,
            size_px=self.size_px,
            volume_shape_zyx=self.volume_shape_zyx,
            labels=(None if self.labels is None else self.labels.copy()),
        )

    # ---------- Distance (class utility) ----------
    @staticmethod
    def distance(a: "Slice", b: "Slice") -> float:
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

        def color_from_id(k: int) -> Tuple[int,int,int]:
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
            if   i == 0: r,g,b = val,t,p
            elif i == 1: r,g,b = q,val,p
            elif i == 2: r,g,b = p,val,t
            elif i == 3: r,g,b = p,q,val
            elif i == 4: r,g,b = t,p,val
            else:        r,g,b = val,p,q
            return int(r*255), int(g*255), int(b*255)

        lut = {int(k): color_from_id(int(k)) for k in uniques}
        a_val = int(np.clip(alpha, 0, 1) * 255)
        for k, (r,g,b) in lut.items():
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
        self._global_lo: Optional[float] = None   # was _global_lo_hi
        self._global_hi: Optional[float] = None   # was _global_lo_hi
        self._is_normalized: bool = False

    def _set_volume(self, arr_zyx: np.ndarray, spacing_zyx: Tuple[float,float,float] = (1.0,1.0,1.0)):
        if arr_zyx.ndim != 3:
            raise ValueError("Volume must be 3D")
        self._vol = _Vol(arr=arr_zyx, spacing=spacing_zyx)
        lo, hi = _compute_lo_hi(arr_zyx, percentiles=(0.5, 99.5))
        self._global_lo, self._global_hi = float(lo), float(hi)
        self._is_normalized = False

    
    def is_valid_slice(
        self,
        sl: Slice,
        *,
        ratio_threshold: float = 0.10,
        value_threshold_pct: float = 0.10,
    ) -> bool:
        """
        Decide whether a slice contains enough tissue vs background.

        Uses global volume bounds:
        - If volume is raw: (_global_lo, _global_hi) computed once in _set_volume.
        - If volume is normalized: (_global_lo, _global_hi) = (0, 1).

        Threshold is: thr = lo + value_threshold_pct * (hi - lo).
        """
        if self._global_lo is None or self._global_hi is None:
            raise RuntimeError("Global intensity bounds not initialized")

        lo, hi = self._global_lo, self._global_hi
        thr = lo + value_threshold_pct * (hi - lo)

        img = sl.image
        mask = img > thr
        return (mask.sum() / img.size) > ratio_threshold
    
    # ---------- tiny internals now in the class ----------
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build map_coordinates coords for a slice plane.
        Returns (coords, center_zyx). coords: (3,H,W) ordered (Z,Y,X).
        """
        Z, Y, X = self._vol.arr.shape
        n = self._unit(normal_xyz.astype(np.float64))
        # basis {u, v, n} in voxel space (X,Y,Z)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = self._unit(ref - np.dot(ref, n) * n)
        v = np.cross(n, u)

        th = math.radians(rotation_deg)
        u_r =  math.cos(th) * u + math.sin(th) * v
        v_r = -math.sin(th) * u + math.cos(th) * v

        H = W = int(size)
        half = (H - 1) / 2.0
        su = (np.arange(W) - half) * pixel
        sv = (np.arange(H) - half) * pixel
        UU, VV = np.meshgrid(su, sv)

        center_xyz = np.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], dtype=np.float64)
        P = center_xyz + depth * n + UU[..., None] * u_r + VV[..., None] * v_r  # (H,W,3)
        coords = np.stack([P[..., 2], P[..., 1], P[..., 0]], axis=0)  # (3,H,W)
        return coords, np.array([center_xyz[2], center_xyz[1], center_xyz[0]], dtype=np.float64)

    # --------------- public API ---------------
    def get_dimension(self) -> Tuple[int, int, int]:
        return tuple(self._vol.arr.shape)  # (Z,Y,X)
    
    def normalize_volume(
        self,
        *,
        percentiles: tuple[float, float] = (0.5, 99.5),
    ) -> None:
        arr = self._vol.arr.astype(np.float32, copy=False)

        lo, hi = np.percentile(arr, percentiles)
        scale = max(hi - lo, 1e-12)  # epsilon guard

        arr01 = (arr - float(lo)) / float(scale)
        np.clip(arr01, 0.0, 1.0, out=arr01)

        # write back and mark normalized
        self._vol = _Vol(arr=arr01.astype(np.float32, copy=False), spacing=self._vol.spacing)
        self._global_lo, self._global_hi = 0.0, 1.0
        self._is_normalized = True

    def is_normalized(self) -> bool:
        """Return True if the volume has been normalized to [0,1]."""
        return bool(self._is_normalized)
    
    def get_global_intensity_bounds(self) -> Tuple[float, float]:
        return (self._global_lo, self._global_hi)

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
        V = self._vol.arr
        Z, Y, X = V.shape

        n = np.asarray(normal, dtype=np.float64)
        coords, _ = self._build_plane_coords(n, depth, rotation, size, pixel)

        order = 1 if linear_interp and np.issubdtype(V.dtype, np.floating) else 0
        img = map_coordinates(V, coords, order=order, mode="nearest").astype(np.float32)

        labels = None
        if include_annotation:
            if hasattr(self, "_sample_annotation_slice"):
                labels = getattr(self, "_sample_annotation_slice")(coords)
            elif annotation_helper is not None:
                labels = annotation_helper.sample_labels(coords)

        return Slice(
            image=img,
            normal_xyz_unit=tuple(self._unit(n).tolist()),
            depth_vox=float(depth),
            rotation_deg=float(rotation),
            pixel_step_vox=float(pixel),
            size_px=int(size),
            volume_shape_zyx=(Z, Y, X),
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
        coords_zyx: (3,H,W) in (Z,Y,X) index order from _build_plane_coords/get_slice
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
        self._annot_helper = AnnotationHelper(cache_dir=self.cache_dir, resolution_um=self.res_um)
        self.load()

    def load(self):
        arr, _ = self._rc.get_template_volume()   # (Z,Y,X)
        self._set_volume(arr.astype(np.float32, copy=False), spacing_zyx=(1.0, 1.0, 1.0))

    # Used by VolumeHelper.get_slice when include_annotation=True
    def _sample_annotation_slice(self, coords_zyx: np.ndarray) -> np.ndarray:
        return self._annot_helper.sample_labels(coords_zyx)

    def _find_cached(self, filename: str) -> Path:
        matches = list(self.cache_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Could not locate cached file: {filename} under {self.cache_dir}")
        return matches[0]


# ------------------------- Concrete: NIfTI -------------------------

class NiftiVolume(VolumeHelper):
    """Loads a NIfTI (.nii / .nii.gz) volume as (Z,Y,X) float32."""
    def __init__(self, nifti_path: str | Path):
        super().__init__()
        self.path = Path(nifti_path)
        self.load()

    def load(self):
        img = nib.load(str(self.path))
        data_xyz = img.get_fdata(dtype=np.float32)  # nibabel gives (X,Y,Z)
        # arr_zyx = np.transpose(data_xyz, (2, 1, 0))  # -> (Z,Y,X)
        zooms = img.header.get_zooms()[:3]           # (X,Y,Z) sizes, often in mm
        spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        self._set_volume(data_xyz, spacing_zyx=spacing_zyx)
