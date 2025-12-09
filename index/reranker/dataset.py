# index/reranker/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import math
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from volume.volume_helper import AllenVolume, NiftiVolume, VolumeHelper, Slice


# ---------------------- CSV-based dataset (unchanged) ----------------------


@dataclass(frozen=True)
class PairSample:
    query_path: Path
    candidate_path: Path
    target: float


class ImagePairDataset(Dataset):
    """
    Generic image-pair dataset for reranker training (from CSV).

    Expects a CSV with at least the columns:
      - query_path
      - candidate_path
      - target   (float regression label, e.g. relevance in [0,1])

    Paths can be absolute, or relative to `root`.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        root: str | Path | None = None,
        query_col: str = "query_path",
        candidate_col: str = "candidate_path",
        target_col: str = "target",
        grayscale: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else None
        self.query_col = query_col
        self.candidate_col = candidate_col
        self.target_col = target_col
        self.grayscale = grayscale

        self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if self.root is not None and not p.is_absolute():
            p = self.root / p
        return p

    def _load_image(self, path: Path) -> np.ndarray:
        mode = "L" if self.grayscale else "RGB"
        img = Image.open(path).convert(mode)
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        q_path = self._resolve(str(row[self.query_col]))
        c_path = self._resolve(str(row[self.candidate_col]))
        target = float(row[self.target_col])

        q_img = self._load_image(q_path)
        c_img = self._load_image(c_path)

        return {
            "q_img": q_img,
            "c_img": c_img,
            "target": target,
        }


# ---------------------- Volume-based dataset (updated) ----------------------

class VolumePairDataset(Dataset):
    """
    On-the-fly pair generator from two registered volumes (Allen + real).

    For each sample:
      - Sample a random plane (normal, depth, rotation).
      - Choose query volume: Allen or real.
      - With probability `near_prob`:
          * Candidate = same pose in the *other* volume (near, spatially aligned)
          * target ~ 1 (via distance->target mapping)
        Else:
          * Candidate = slice from a far pose (large center distance)
          * target decays with spatial distance in [0,1].

    Multi-scale behaviour:
      - After sampling slices, optionally apply a random crop, so the model
        sees full slices and patches of different sizes. Crops are then
        resized back to (slice_size, slice_size).

    Augmentation:
      - Basic geometric + photometric augmentations are applied independently
        to query and candidate images to improve robustness to user rotations,
        flips, and mild appearance changes.
    """

    def __init__(
        self,
        allen: AllenVolume,
        real: NiftiVolume,
        *,
        n_samples: int = 50000,
        slice_size: int = 224,
        pixel_step_vox: float = 1.0,
        near_prob: float = 0.5,
        max_resample_tries: int = 20,
        min_tissue_ratio: float = 0.10,
        min_far_fraction: float = 0.3,
        length_scale_fraction: float = 0.25,
        # multi-scale + augmentation flags
        enable_crops: bool = True,
        crop_prob: float = 0.8,
        enable_augment: bool = True,
    ):
        self.allen = allen
        self.real = real
        self.n_samples = int(n_samples)
        self.slice_size = int(slice_size)
        self.pixel_step_vox = float(pixel_step_vox)
        self.near_prob = float(near_prob)
        self.max_resample_tries = int(max_resample_tries)
        self.min_tissue_ratio = float(min_tissue_ratio)
        self.min_far_fraction = float(min_far_fraction)
        self.length_scale_fraction = float(length_scale_fraction)

        self.enable_crops = bool(enable_crops)
        self.crop_prob = float(crop_prob)
        self.enable_augment = bool(enable_augment)

        # Make sure volumes are normalized; if not, normalize to [0,1].
        if not self.allen.is_normalized():
            self.allen.normalize_volume()
        if not self.real.is_normalized():
            self.real.normalize_volume()

        # Use Allen shape as reference (volumes are registered).
        self.Z, self.Y, self.X = self.allen.get_dimension()
        diag = float(math.sqrt(self.Z**2 + self.Y**2 + self.X**2))  # voxel diag
        self.volume_diag = diag
        self.min_far_dist = self.min_far_fraction * diag
        self.length_scale = self.length_scale_fraction * diag

    def __len__(self) -> int:
        return self.n_samples

    # --------- random pose helpers ---------

    @staticmethod
    def _random_unit_vector() -> Tuple[float, float, float]:
        v = np.random.normal(size=(3,))
        n = float(np.linalg.norm(v))
        v = v / (n + 1e-12)
        return float(v[0]), float(v[1]), float(v[2])

    def _random_depth(self) -> float:
        r = 0.5 * float(min(self.Z, self.Y, self.X))
        return float(np.random.uniform(-r, r))

    @staticmethod
    def _random_rotation_deg() -> float:
        return float(np.random.uniform(0.0, 360.0))

    def _sample_valid_slice(
        self,
        vol: VolumeHelper,
        normal: Tuple[float, float, float],
        depth: float,
        rotation: float,
    ) -> Slice | None:
        for _ in range(self.max_resample_tries):
            sl = vol.get_slice(
                normal=normal,
                depth=depth,
                rotation=rotation,
                size=self.slice_size,
                pixel=self.pixel_step_vox,
                include_annotation=False,
            ).normalized()
            if vol.is_valid_slice(sl, ratio_threshold=self.min_tissue_ratio):
                return sl
        return None

    @staticmethod
    def _center_distance(a: Slice, b: Slice) -> float:
        ax, ay, az = a.center_xyz_vox
        bx, by, bz = b.center_xyz_vox
        return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))

    def _distance_to_target(self, dist: float) -> float:
        """
        Map a spatial distance to a target in [0,1], decaying with distance.

        We use: target = exp(- dist / length_scale).
        """
        return float(math.exp(-dist / (self.length_scale + 1e-12)))

    # --------- multi-scale cropping helpers ---------

    def _random_crop_params(self) -> Tuple[float, float, float, float]:
        """
        Sample (cx, cy, rw, rh) where:
          - cx, cy in [0,1]  (normalized center)
          - rw, rh in (0,1] (relative size)
        We use 3 scale bands: small, medium, large.
        """
        band = random.random()
        if band < 0.3:        # small patch: 10–30%
            rw = np.random.uniform(0.1, 0.3)
            rh = np.random.uniform(0.1, 0.3)
        elif band < 0.7:      # medium: 30–70%
            rw = np.random.uniform(0.3, 0.7)
            rh = np.random.uniform(0.3, 0.7)
        else:                 # large: 70–100%
            rw = np.random.uniform(0.7, 1.0)
            rh = np.random.uniform(0.7, 1.0)

        cx = np.random.uniform(0.0, 1.0)
        cy = np.random.uniform(0.0, 1.0)
        return float(cx), float(cy), float(rw), float(rh)

    def _maybe_apply_random_crop_pair(
        self,
        sl_q: Slice,
        sl_c: Slice,
    ) -> Tuple[Slice, Slice]:
        """
        With probability crop_prob, apply the SAME crop parameters to query
        and candidate, to keep them looking at the same relative region.

        Otherwise, return the original slices (full slice).
        """
        if not self.enable_crops or random.random() > self.crop_prob:
            return sl_q, sl_c

        cx, cy, rw, rh = self._random_crop_params()
        try:
            sl_q_crop = sl_q.crop_norm(cx, cy, rw, rh)
            sl_c_crop = sl_c.crop_norm(cx, cy, rw, rh)
            return sl_q_crop, sl_c_crop
        except Exception:
            return sl_q, sl_c

    # --------- augmentation helpers ---------

    @staticmethod
    def _random_geo_augment(img: np.ndarray) -> np.ndarray:
        """
        Basic geometric augmentations:
          - random 90° rotations
          - random horizontal/vertical flips
        """
        out = img
        k = random.randint(0, 3)
        if k:
            out = np.rot90(out, k, axes=(0, 1))
        if random.random() < 0.5:
            out = np.flip(out, axis=1)
        if random.random() < 0.5:
            out = np.flip(out, axis=0)
        return out.copy()

    @staticmethod
    def _random_photometric_augment(img: np.ndarray) -> np.ndarray:
        """
        Simple brightness/contrast jitter + small Gaussian noise.
        Assumes img in [0,1].
        """
        out = img.astype(np.float32, copy=False)

        b = np.random.uniform(-0.1, 0.1)   # brightness
        c = np.random.uniform(0.9, 1.1)    # contrast

        out = c * out + b
        if np.random.rand() < 0.5:
            noise = np.random.normal(scale=0.02, size=out.shape).astype(np.float32)
            out = out + noise

        out = np.clip(out, 0.0, 1.0)
        return out

    def _augment_pair(
        self,
        q_img: np.ndarray,
        c_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_augment:
            return q_img, c_img
        q = self._random_geo_augment(q_img)
        c = self._random_geo_augment(c_img)
        q = self._random_photometric_augment(q)
        c = self._random_photometric_augment(c)
        return q, c

    def _resize_to_slice_size(self, img: np.ndarray) -> np.ndarray:
        """
        Resize img (H,W) in [0,1] to (slice_size, slice_size) using PIL bilinear.
        """
        H, W = img.shape
        if H == self.slice_size and W == self.slice_size:
            return img.astype(np.float32, copy=False)

        # to uint8 for PIL
        arr_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_uint8, mode="L")
        pil_resized = pil.resize((self.slice_size, self.slice_size), resample=Image.BILINEAR)
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        return arr

    # --------- main sampling logic ---------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) pose
        normal = self._random_unit_vector()
        depth = self._random_depth()
        rotation = self._random_rotation_deg()

        # 2) choose query volume
        q_from_allen = bool(random.random() < 0.5)
        vol_q = self.allen if q_from_allen else self.real
        vol_other = self.real if q_from_allen else self.allen

        # 3) query slice
        sl_q = self._sample_valid_slice(vol_q, normal, depth, rotation)
        if sl_q is None:
            return self.__getitem__((idx + 1) % self.n_samples)

        # 4) near / far candidate
        if random.random() < self.near_prob:
            sl_c = self._sample_valid_slice(vol_other, normal, depth, rotation)
            if sl_c is None:
                sl_c, dist_base = self._sample_far(sl_q, vol_other)
            else:
                dist_base = self._center_distance(sl_q, sl_c)
        else:
            sl_c, dist_base = self._sample_far(sl_q, vol_other)

        # 5) multi-scale cropping
        sl_q, sl_c = self._maybe_apply_random_crop_pair(sl_q, sl_c)

        # 6) distance -> target
        target = self._distance_to_target(dist_base)

        # 7) images in [0,1]
        q_img = sl_q.image.astype(np.float32, copy=False)
        c_img = sl_c.image.astype(np.float32, copy=False)
        q_img = np.clip(q_img, 0.0, 1.0)
        c_img = np.clip(c_img, 0.0, 1.0)

        # 8) augment
        q_img, c_img = self._augment_pair(q_img, c_img)

        # 9) resize to fixed size so DataLoader can stack
        q_img = self._resize_to_slice_size(q_img)
        c_img = self._resize_to_slice_size(c_img)

        return {
            "q_img": q_img,
            "c_img": c_img,
            "target": target,
        }

    def _sample_far(
        self,
        sl_q: Slice,
        vol_candidate: VolumeHelper,
    ) -> Tuple[Slice, float]:
        last_valid: Slice | None = None
        last_dist: float = 0.0

        for _ in range(self.max_resample_tries):
            n2 = self._random_unit_vector()
            d2 = self._random_depth()
            r2 = self._random_rotation_deg()
            sl_c = self._sample_valid_slice(vol_candidate, n2, d2, r2)
            if sl_c is None:
                continue
            dist = self._center_distance(sl_q, sl_c)
            last_valid = sl_c
            last_dist = dist
            if dist >= self.min_far_dist:
                return sl_c, dist

        if last_valid is not None:
            return last_valid, last_dist
        return sl_q, 0.0