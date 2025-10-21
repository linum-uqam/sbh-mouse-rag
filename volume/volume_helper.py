from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import math
import numpy as np
from scipy.ndimage import map_coordinates
from allensdk.core.reference_space_cache import ReferenceSpaceCache  # AllenVolume
import nibabel as nib  # NiftiVolume
import matplotlib.pyplot as plt


# -------------------- Slice object --------------------

@dataclass(frozen=True)
class Slice:
    image: np.ndarray                         # (H,W) float32
    normal_xyz_unit: Tuple[float, float, float]
    depth_vox: float
    rotation_deg: float
    pixel_step_vox: float
    size_px: int
    volume_shape_zyx: Tuple[int, int, int]

    def save(self, path: str | Path, title: str | None = None, dpi: int = 200) -> Path:
        """
        Save this slice image to a file.
        NOTE: this method assumes the parent directory exists; create it in your usage code.
        """

        path = Path(path)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.image, cmap="gray")
        if title:
            ax.set_title(title)
        ax.axis("off")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path

    def crop_norm(
        self,
        cx: float, cy: float,   # center in [0,1] (x then y)
        rw: float, rh: float,   # relative width/height in (0,1]
        *,
        clamp: bool = True,
    ) -> "Slice":
        """
        Deterministic crop using normalized args:
          - center (cx, cy) in [0,1] relative to image (x = horiz, y = vert)
          - relative size (rw, rh) in (0,1] of image width/height
        If clamp=True, crop is clipped to image bounds; else raises if OOB.
        Returns a NEW Slice with the cropped image and same pose metadata.
        """
        H, W = self.image.shape
        w = max(1, int(round(W * rw)))
        h = max(1, int(round(H * rh)))

        # center in pixel coords
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
                # degenerate after clamp -> force at least 1x1 within bounds
                x0, y0 = min(W-1, x0), min(H-1, y0)
                x1, y1 = min(W, x0+1), min(H, y0+1)
        else:
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                raise ValueError("Crop window is out of bounds and clamp=False.")

        sub = self.image[y0:y1, x0:x1]

        return Slice(
            image=sub.copy(),
            normal_xyz_unit=self.normal_xyz_unit,
            depth_vox=self.depth_vox,
            rotation_deg=self.rotation_deg,
            pixel_step_vox=self.pixel_step_vox,
            size_px=min(sub.shape[0], sub.shape[1]),
            volume_shape_zyx=self.volume_shape_zyx,
        )

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

    def _set_volume(self, arr_zyx: np.ndarray, spacing_zyx: Tuple[float,float,float] = (1.0,1.0,1.0)):
        if arr_zyx.ndim != 3:
            raise ValueError("Volume must be 3D")
        self._vol = _Vol(arr=arr_zyx, spacing=spacing_zyx)

    # --------------- public API ---------------

    def get_dimension(self) -> Tuple[int, int, int]:
        return tuple(self._vol.arr.shape)  # (Z,Y,X)

    def get_slice(
        self,
        normal: tuple[float, float, float],
        depth: float,
        rotation: float = 0.0,
        size: int = 512,
        pixel: float = 1.0,
        linear_interp: bool = True,
    ) -> Slice:
        """
        Returns a Slice object with the sampled image and pose metadata.
        - normal: (X,Y,Z) vector (unit-normalized internally)
        - depth : offset from center (voxels, along +normal)
        - rotation: in-plane (deg), CCW around normal
        - size: H=W output pixels
        - pixel: sampling step in voxels on the plane
        - linear_interp: True->order=1, False->order=0 (nearest)
        """
        V = self._vol.arr
        Z, Y, X = V.shape

        n = np.asarray(normal, dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)

        # basis {u, v, n} in voxel space (X,Y,Z)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, n) * n; u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)

        th = math.radians(rotation)
        u_r =  math.cos(th) * u + math.sin(th) * v
        v_r = -math.sin(th) * u + math.cos(th) * v

        H = W = int(size)
        half = (H - 1) / 2.0
        su = (np.arange(W) - half) * pixel
        sv = (np.arange(H) - half) * pixel
        UU, VV = np.meshgrid(su, sv)

        center = np.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], dtype=np.float64)
        P = center + depth * n + UU[..., None] * u_r + VV[..., None] * v_r  # (H,W,3)

        coords = np.stack([P[..., 2], P[..., 1], P[..., 0]], axis=0)  # (3,H,W)
        order = 1 if linear_interp else 0
        img = map_coordinates(V, coords, order=order, mode="nearest").astype(np.float32)

        # Normalize if float volume (nice for display)
        if np.issubdtype(V.dtype, np.floating):
            lo, hi = np.percentile(img, [0.5, 99.5])
            img = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)

        return Slice(
            image=img,
            normal_xyz_unit=(float(n[0]), float(n[1]), float(n[2])),
            depth_vox=float(depth),
            rotation_deg=float(rotation),
            pixel_step_vox=float(pixel),
            size_px=int(size),
            volume_shape_zyx=(Z, Y, X),
        )

# --------------- oriented distance between two slices ---------------

def slice_distance(a: Slice, b: Slice) -> float:
    """
    Oriented slice distance (voxels) using only data from the slices:
      dist = || d_a*n_a - d_b*n_b ||  +  k * theta
      where theta = angle(n_a, n_b) in radians, and
            k = 0.5 * max(max_dim_a, max_dim_b).
    """
    n1 = np.asarray(a.normal_xyz_unit, float)
    n2 = np.asarray(b.normal_xyz_unit, float)
    d1 = float(a.depth_vox)
    d2 = float(b.depth_vox)

    # positional term
    pos = float(np.linalg.norm(d1 * n1 - d2 * n2))

    # angular term
    dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    theta = math.acos(dot)  # [0, pi], oriented

    # scale by volume size from either slice
    Za, Ya, Xa = a.volume_shape_zyx
    Zb, Yb, Xb = b.volume_shape_zyx
    k = 0.5 * max(max(Xa, Ya, Za), max(Xb, Yb, Zb))

    return pos + k * theta



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
        self.load()

    def load(self):
        arr, _ = self._rc.get_template_volume()   # (Z,Y,X)
        self._set_volume(arr.astype(np.float32, copy=False), spacing_zyx=(1.0, 1.0, 1.0))

    def _download_brain(self) -> Path:
        self._rc.get_template_volume()
        return self._find_cached(f"average_template_{self.res_um}.nrrd")

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
        # arr_zyx = np.transpose(data_xyz, (2, 1, 0))  # -> (Z,Y,X)  ✅ fix
        zooms = img.header.get_zooms()[:3]           # (X,Y,Z) sizes, often in mm
        spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        self._set_volume(data_xyz, spacing_zyx=spacing_zyx)
