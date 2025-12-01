# dataset/builder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Any

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
    rows_collected: int = 0
    planes_skipped: int = 0
    crop_attempts: int = 0
    crops_valid: int = 0
    plane_attempts: int = 0  # number of plane draws

    def postfix(self) -> dict:
        rows = int(self.rows_collected)
        crop_invalid = max(0, self.crop_attempts - self.crops_valid)
        attempts = rows + self.planes_skipped + crop_invalid
        fails = attempts - rows
        pass_pct = 100.0 * rows / max(1, attempts)

        return dict(
            rows=rows,
            attempts=attempts,
            fails=fails,
            succeful=f"{pass_pct:.1f}%",
            planes=self.plane_attempts,
            crops=self.crops_valid,
        )


@dataclass
class PlaneSampler:
    vol_shape_zyx: Tuple[int, int, int]

    def sample_plane(self) -> Tuple[Vec3, float, float]:
        """
        Sample a random unit normal + depth inside volume box + in-plane rotation.
        """
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
        """
        Compute signed depth bounds along n for the axis-aligned volume box.
        """
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
    min_frac: float
    max_frac: float
    n_bins: int = 4       # how many distinct scale bands (small/med/large...)
    jitter: float = 0.15  # how much to jitter around the bin center (15%)

    def sample_crop_params(self) -> Tuple[float, float, float, float]:
        """
        Sample normalized crop parameters (cx, cy, rw, rh) in [0,1].

        Strategy:
          - Split [min_frac, max_frac] into `n_bins` bins.
          - Pick a bin center for width & height (independently).
          - Jitter each center by up to ±`jitter` proportion (clamped to [min,max]).
          - Sample center (cx, cy) uniformly in [0,1].
        """
        min_f = float(self.min_frac)
        max_f = float(self.max_frac)
        if not (0.0 < min_f <= max_f <= 1.0):
            raise ValueError("min_frac/max_frac must be in (0,1] and min_frac <= max_frac")

        # Bin centers between min and max (small → large crops)
        centers = np.linspace(min_f, max_f, num=self.n_bins)

        # Pick a center independently for width and height
        c_w = float(random.choice(centers))
        c_h = float(random.choice(centers))

        def jitter_one(c: float) -> float:
            # Allow +/- jitter on the center, but clamp to [min_f, max_f]
            low = max(min_f, c * (1.0 - self.jitter))
            high = min(max_f, c * (1.0 + self.jitter))
            if high <= low:
                return c
            return random.uniform(low, high)

        rw = jitter_one(c_w)
        rh = jitter_one(c_h)

        # Random crop center in normalized coordinates
        cx = random.uniform(0.0, 1.0)
        cy = random.uniform(0.0, 1.0)

        return float(cx), float(cy), float(rw), float(rh)


@dataclass
class VolumePair:
    """
    Convenience wrapper around Allen + Real volumes.
    """
    allen: AllenVolume
    real: NiftiVolume

    @classmethod
    def from_paths(cls, allen_cache_dir: str, allen_res_um: int, real_path: str) -> "VolumePair":
        allen = AllenVolume(cache_dir=allen_cache_dir, resolution_um=allen_res_um)
        real = NiftiVolume(real_path)
        return cls(allen=allen, real=real)

    def sample_pair(
        self,
        normal: Vec3,
        depth: float,
        rotation: float,
        size: int,
    ) -> Tuple[Slice, Slice]:
        a = self.allen.get_slice(
            normal=normal,
            depth=depth,
            rotation=rotation,
            size=size,
        )
        r = self.real.get_slice(
            normal=normal,
            depth=depth,
            rotation=rotation,
            size=size,
        )
        return a, r


# ============================================================
# Main builder
# ============================================================

class MouseBrainDatasetBuilder:
    """
    Builds a dataset of Allen/Real slices + random crops:
      - CFG.num_slices controls total number of rows (full + crops)
      - For each valid plane, we add:
          * 1 full-slice row
          * up to CFG.max_crop_attempts crop rows (if they are valid)
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

        # RNG seeding
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Output dirs
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Volumes
        self.volumes = VolumePair.from_paths(
            allen_cache_dir=self.cfg.allen_cache_dir,
            allen_res_um=self.cfg.allen_res_um,
            real_path=self.cfg.real_volume_path,
        )

        # Depth bounds / plane sampler
        self._vol_shape_zyx = self.volumes.allen.get_dimension()
        self.plane_sampler = PlaneSampler(self._vol_shape_zyx)
        self.crop_sampler = CropSampler(
            min_frac=self.cfg.min_crop_frac,
            max_frac=self.cfg.max_crop_frac,
        )

        # Stats
        self.stats = DatasetStats()

    # ---------- public API ----------

    def run(self) -> None:
        """
        Main entry: generate cfg.num_slices rows (full + crops) and write:
          - PNGs under cfg.out_dir
          - CSV at cfg.csv_path
        """
        DatasetSchema.init_csv(self.cfg.csv_path)

        target = self.cfg.num_slices
        attempts = 0

        pbar = tqdm(
            total=target,
            desc="Generating slices",
            dynamic_ncols=True,
            leave=True,
        )

        while self.stats.rows_collected < target:
            attempts += 1
            self.stats.plane_attempts += 1

            rows_before = self.stats.rows_collected

            # 1) sample a random plane
            n_vec, depth, rotation = self.plane_sampler.sample_plane()

            # 2) sample slices for this plane
            a_slice, r_slice = self.volumes.sample_pair(
                normal=n_vec,
                depth=depth,
                rotation=rotation,
                size=self.cfg.slice_size,
            )

            # 3) validate full Allen slice
            if not self.volumes.allen.is_valid_slice(a_slice):
                self.stats.planes_skipped += 1
                pbar.set_postfix(self.stats.postfix())
                continue

            rows_left = target - self.stats.rows_collected
            if rows_left <= 0:
                break

            # 4) add full slice
            row_id = self.stats.rows_collected
            self._add_full_slice(
                row_id=row_id,
                a_slice=a_slice,
                r_slice=r_slice,
                n_vec=n_vec,
                depth=depth,
                rotation=rotation,
            )
            self.stats.rows_collected += 1
            rows_left = target - self.stats.rows_collected

            # 5) add crops (up to cfg.max_crop_attempts, but not exceeding rows_left)
            if rows_left > 0 and self.cfg.max_crop_attempts > 0:
                added_crops = self._add_crops_for_slice(
                    row_id=row_id,
                    a_slice=a_slice,
                    r_slice=r_slice,
                    n_vec=n_vec,
                    depth=depth,
                    rotation=rotation,
                    max_new_rows=rows_left,
                )
                self.stats.rows_collected += added_crops

            rows_after = self.stats.rows_collected
            rows_added = rows_after - rows_before
            if rows_added > 0:
                pbar.update(rows_added)

            pbar.set_postfix(self.stats.postfix())

        pbar.close()

    # ---------- internal helpers ----------

    @staticmethod
    def _save_img(path: Path, arr: np.ndarray) -> None:
        plt.imsave(path, arr, cmap="gray")

    def _add_full_slice(
        self,
        row_id: int,
        a_slice: Slice,
        r_slice: Slice,
        n_vec: Vec3,
        depth: float,
        rotation: float,
    ) -> None:
        # File names: "00000_a.png" and "00000_r.png"
        allen_path = self.cfg.out_dir / f"{row_id:05d}_a.png"
        real_path = self.cfg.out_dir / f"{row_id:05d}_r.png"

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

    def _add_crops_for_slice(
        self,
        row_id: int,
        a_slice: Slice,
        r_slice: Slice,
        n_vec: Vec3,
        depth: float,
        rotation: float,
        max_new_rows: int,
    ) -> int:
        """
        Try up to cfg.max_crop_attempts random crops for this plane.
        A crop is accepted if the Allen crop is valid.
        """
        cfg = self.cfg
        added = 0

        for attempt in range(cfg.max_crop_attempts):
            if added >= max_new_rows:
                break

            self.stats.crop_attempts += 1

            cx, cy, rw, rh = self.crop_sampler.sample_crop_params()

            a_crop = a_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)
            r_crop = r_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)

            # Use Allen volume-based foreground check
            if not self.volumes.allen.is_valid_slice(a_crop):
                continue

            crop_idx = added
            added += 1
            self.stats.crops_valid += 1

            # File names: "00000_a_crop0.png" and "00000_r_crop0.png"
            allen_path = cfg.out_dir / f"{row_id:05d}_a_crop{crop_idx}.png"
            real_path = cfg.out_dir / f"{row_id:05d}_r_crop{crop_idx}.png"

            if cfg.save_images:
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
            DatasetSchema.append_row(cfg.csv_path, row)

        return added
