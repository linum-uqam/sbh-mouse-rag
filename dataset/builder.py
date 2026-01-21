# dataset/builder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from volume.volume_helper import AllenVolume, NiftiVolume, Slice

from .schema import DatasetRow, DatasetSchema, Vec3
from .config import DatasetConfig


# ============================================================
# Helper classes
# ============================================================

@dataclass
class DatasetStats:
    full_written: int = 0
    crops_written: int = 0
    planes_resampled: int = 0
    crop_attempts: int = 0
    crop_failures: int = 0

    def postfix(self) -> dict:
        return dict(
            full=self.full_written,
            crops=self.crops_written,
            planes_resampled=self.planes_resampled,
            crop_attempts=self.crop_attempts,
            crop_failures=self.crop_failures,
        )


@dataclass
class PlaneSampler:
    vol_shape_zyx: Tuple[int, int, int]

    def sample_plane(self) -> Tuple[Vec3, float, float]:
        n_vec = self._random_normal()
        dmin, dmax = self._depth_bounds(n_vec)
        depth = float(np.random.uniform(dmin, dmax))
        rotation = random.random() * 360.0
        return n_vec, depth, rotation

    @staticmethod
    def _random_normal() -> Vec3:
        v = np.random.normal(size=3).astype(np.float64)
        n = v / (np.linalg.norm(v) + 1e-12)
        return (float(n[0]), float(n[1]), float(n[2]))

    def _depth_bounds(self, n_xyz_unit: Vec3) -> Tuple[float, float]:
        Z, Y, X = self.vol_shape_zyx
        cx, cy, cz = ((X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0)
        n = np.asarray(n_xyz_unit, dtype=np.float64)

        corners: List[np.ndarray] = []
        for dx in (-cx, +cx):
            for dy in (-cy, +cy):
                for dz in (-cz, +cz):
                    corners.append(np.array([dx, dy, dz], dtype=np.float64))

        proj = [float(np.dot(c, n)) for c in corners]
        return min(proj), max(proj)


@dataclass
class CropSampler:
    """
    Crop bins are fixed at 3 (large/medium/small) + full slice is handled separately.
    """
    min_frac: float
    max_frac: float
    jitter: float = 0.15  # +/- around bin center

    def __post_init__(self) -> None:
        min_f = float(self.min_frac)
        max_f = float(self.max_frac)
        if not (0.0 < min_f <= max_f <= 1.0):
            raise ValueError("min_frac/max_frac must be in (0,1] and min_frac <= max_frac")

        # 3 crop bins => 4 edges
        self._edges = np.linspace(min_f, max_f, num=4).astype(np.float64)

    def sample_crop_params_for_bin(self, bin_idx: int) -> Tuple[float, float, float, float]:
        """
        bin_idx:
          0 -> large
          1 -> medium
          2 -> small

        We map indices so that "0 is large" (intuitive with crop0 naming).
        Internally edges go min->max, so we invert.
        """
        if bin_idx not in (0, 1, 2):
            raise ValueError("bin_idx must be 0 (large), 1 (medium), or 2 (small)")

        # Invert so 0=large uses the largest interval, 2=small uses smallest.
        inv = 2 - bin_idx
        low = float(self._edges[inv])
        high = float(self._edges[inv + 1])
        center = 0.5 * (low + high)

        def sample_size() -> float:
            # jitter center but clamp to [low, high]
            j_low = max(low, center * (1.0 - self.jitter))
            j_high = min(high, center * (1.0 + self.jitter))
            if j_high <= j_low:
                return center
            return random.uniform(j_low, j_high)

        rw = float(sample_size())
        rh = float(sample_size())

        cx = random.uniform(0.0, 1.0)
        cy = random.uniform(0.0, 1.0)
        return cx, cy, rw, rh


@dataclass
class VolumePair:
    allen: AllenVolume
    real: NiftiVolume

    @classmethod
    def from_paths(cls, allen_cache_dir: str, allen_res_um: int, real_path: str) -> "VolumePair":
        allen = AllenVolume(cache_dir=allen_cache_dir, resolution_um=allen_res_um)
        real = NiftiVolume(real_path)
        return cls(allen=allen, real=real)

    def sample_pair(self, normal: Vec3, depth: float, rotation: float, size: int) -> Tuple[Slice, Slice]:
        a = self.allen.get_slice(normal=normal, depth=depth, rotation=rotation, size=size)
        r = self.real.get_slice(normal=normal, depth=depth, rotation=rotation, size=size)
        return a, r


# ============================================================
# Main builder
# ============================================================

class MouseBrainDatasetBuilder:
    """
    Fixed 4 bins:
      - full slice
      - large crop
      - medium crop
      - small crop

    If cfg.num_slices = N, we generate:
      - N full slices
      - N large crops
      - N medium crops
      - N small crops
    Total rows = 4N (and images likewise, if save_images=True).
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.volumes = VolumePair.from_paths(
            allen_cache_dir=self.cfg.allen_cache_dir,
            allen_res_um=self.cfg.allen_res_um,
            real_path=self.cfg.real_volume_path,
        )

        self._vol_shape_zyx = self.volumes.allen.get_dimension()
        self.plane_sampler = PlaneSampler(self._vol_shape_zyx)

        self.crop_sampler = CropSampler(
            min_frac=self.cfg.min_crop_frac,
            max_frac=self.cfg.max_crop_frac,
        )

        self.stats = DatasetStats()

    @staticmethod
    def _save_img(path: Path, arr: np.ndarray) -> None:
        plt.imsave(path, arr, cmap="gray")

    def run(self) -> None:
        DatasetSchema.init_csv(self.cfg.csv_path)

        N = int(self.cfg.num_slices)
        total_rows = 4 * N

        # Store plane parameters per contiguous plane_id
        planes: List[Tuple[Vec3, float, float]] = []

        pbar = tqdm(
            total=total_rows,
            desc="Generating dataset (full + 3 crop bins)",
            dynamic_ncols=True,
            leave=True,
        )

        # ----------------------------
        # Phase A: N valid full slices
        # ----------------------------
        for plane_id in range(N):
            while True:
                n_vec, depth, rotation = self.plane_sampler.sample_plane()
                a_slice, r_slice = self.volumes.sample_pair(
                    normal=n_vec,
                    depth=depth,
                    rotation=rotation,
                    size=self.cfg.slice_size,
                )

                if self.volumes.allen.is_valid_slice(a_slice):
                    # accept
                    planes.append((n_vec, float(depth), float(rotation)))
                    self._write_full(plane_id, a_slice, r_slice, n_vec, depth, rotation)
                    self.stats.full_written += 1
                    pbar.update(1)
                    pbar.set_postfix(self.stats.postfix())
                    break

                self.stats.planes_resampled += 1
                pbar.set_postfix(self.stats.postfix())

        # -------------------------------------------
        # Phase B: for each plane, 3 crops (L/M/S)
        # crop_idx is stable:
        #   0 = large, 1 = medium, 2 = small
        # -------------------------------------------
        for plane_id in range(N):
            n_vec, depth, rotation = planes[plane_id]

            # regenerate base slices (no need to keep all in RAM)
            a_slice, r_slice = self.volumes.sample_pair(
                normal=n_vec,
                depth=depth,
                rotation=rotation,
                size=self.cfg.slice_size,
            )

            for crop_idx in (0, 1, 2):
                self._write_one_crop_for_plane(
                    plane_id=plane_id,
                    crop_idx=crop_idx,
                    a_slice=a_slice,
                    r_slice=r_slice,
                    n_vec=n_vec,
                    depth=depth,
                    rotation=rotation,
                )
                self.stats.crops_written += 1
                pbar.update(1)
                pbar.set_postfix(self.stats.postfix())

        pbar.close()

    # ----------------------------
    # Writers
    # ----------------------------

    def _write_full(
        self,
        plane_id: int,
        a_slice: Slice,
        r_slice: Slice,
        n_vec: Vec3,
        depth: float,
        rotation: float,
    ) -> None:
        allen_path = self.cfg.out_dir / f"{plane_id:05d}_a.png"
        real_path = self.cfg.out_dir / f"{plane_id:05d}_r.png"

        if self.cfg.save_images:
            self._save_img(allen_path, a_slice.image)
            self._save_img(real_path, r_slice.image)

        row = DatasetRow(
            allen_path=str(allen_path),
            real_path=str(real_path),
            vector=n_vec,
            depth=float(depth),
            rotation=float(rotation),
            crop_cx=0.5,
            crop_cy=0.5,
            crop_rw=1.0,
            crop_rh=1.0,
            is_crop=0,
        )
        DatasetSchema.append_row(self.cfg.csv_path, row)

    def _write_one_crop_for_plane(
        self,
        plane_id: int,
        crop_idx: int,  # 0=large, 1=medium, 2=small
        a_slice: Slice,
        r_slice: Slice,
        n_vec: Vec3,
        depth: float,
        rotation: float,
    ) -> None:
        """
        Keep trying until we get a valid crop for the requested bin.
        Uses cfg.max_crop_attempts as the attempt cap *per bin per plane*.
        """
        max_attempts = int(self.cfg.max_crop_attempts)
        if max_attempts <= 0:
            raise ValueError("max_crop_attempts must be > 0 to generate fixed crop quotas")

        for _ in range(max_attempts):
            self.stats.crop_attempts += 1

            cx, cy, rw, rh = self.crop_sampler.sample_crop_params_for_bin(crop_idx)

            a_crop = a_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)
            r_crop = r_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)

            if not self.volumes.allen.is_valid_slice(a_crop):
                continue

            allen_path = self.cfg.out_dir / f"{plane_id:05d}_a_crop{crop_idx}.png"
            real_path = self.cfg.out_dir / f"{plane_id:05d}_r_crop{crop_idx}.png"

            if self.cfg.save_images:
                self._save_img(allen_path, a_crop.image)
                self._save_img(real_path, r_crop.image)

            row = DatasetRow(
                allen_path=str(allen_path),
                real_path=str(real_path),
                vector=n_vec,
                depth=float(depth),
                rotation=float(rotation),
                crop_cx=float(cx),
                crop_cy=float(cy),
                crop_rw=float(rw),
                crop_rh=float(rh),
                is_crop=1,
            )
            DatasetSchema.append_row(self.cfg.csv_path, row)
            return

        # If we get here, we failed to find a valid crop within the cap.
        self.stats.crop_failures += 1
        raise RuntimeError(
            f"Failed to generate a valid crop for plane_id={plane_id}, crop_idx={crop_idx} "
            f"within max_crop_attempts={max_attempts}. "
            "Consider increasing --max-crop-attempts or relaxing --min-crop-frac/--max-crop-frac."
        )
