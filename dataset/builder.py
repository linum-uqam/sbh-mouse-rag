from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from volume.volume_helper import AllenVolume, NiftiVolume, Slice

from .schema import DatasetRow, DatasetSchema, Vec3
from .config import DatasetConfig


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

    Aspect ratio is controlled explicitly through (aspect_w, aspect_h):
      - (1, 1) -> square
      - (2, 1) -> wide rectangle
      - (1, 2) -> tall rectangle

    The sampled size parameter corresponds to an approximate square-equivalent side
    (geometric mean size). This keeps crop size bins comparable across aspect ratios.
    """
    min_frac: float
    max_frac: float
    jitter: float = 0.15

    def __post_init__(self) -> None:
        min_f = float(self.min_frac)
        max_f = float(self.max_frac)
        if not (0.0 < min_f <= max_f <= 1.0):
            raise ValueError("min_frac/max_frac must be in (0,1] and min_frac <= max_frac")
        self._edges = np.linspace(min_f, max_f, num=4).astype(np.float64)

    def _sample_size_for_bin(self, bin_idx: int) -> float:
        if bin_idx not in (0, 1, 2):
            raise ValueError("bin_idx must be 0 (large), 1 (medium), or 2 (small)")

        inv = 2 - bin_idx
        low = float(self._edges[inv])
        high = float(self._edges[inv + 1])
        center = 0.5 * (low + high)

        j_low = max(low, center * (1.0 - self.jitter))
        j_high = min(high, center * (1.0 + self.jitter))
        if j_high <= j_low:
            return center
        return random.uniform(j_low, j_high)

    def sample_crop_params_for_bin_and_aspect(
        self,
        bin_idx: int,
        *,
        aspect_w: float,
        aspect_h: float,
    ) -> Tuple[float, float, float, float]:
        if aspect_w <= 0 or aspect_h <= 0:
            raise ValueError("aspect_w and aspect_h must be > 0")

        area_side = float(self._sample_size_for_bin(bin_idx))
        ratio = float(aspect_w) / float(aspect_h)
        ratio_sqrt = math.sqrt(ratio)

        rw = area_side * ratio_sqrt
        rh = area_side / ratio_sqrt

        max_side = max(rw, rh)
        if max_side > 1.0:
            scale = 1.0 / max_side
            rw *= scale
            rh *= scale

        cx = random.uniform(0.0, 1.0)
        cy = random.uniform(0.0, 1.0)
        return cx, cy, float(rw), float(rh)


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


class MouseBrainDatasetBuilder:
    """
    Per plane we generate:
      - 1 full slice
      - 3 crop bins (large/medium/small)
      - for each configured aspect-ratio family

    Total rows = num_slices * (1 + 3 * len(cfg.crop_aspect_ratios))
    """

    CROP_BIN_LABELS = {0: "large", 1: "medium", 2: "small"}

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

    def _iter_crop_aspects(self) -> List[tuple[str, tuple[float, float]]]:
        labels = tuple(self.cfg.crop_aspect_labels)
        ratios = tuple(self.cfg.crop_aspect_ratios)
        if len(labels) != len(ratios):
            raise ValueError("crop_aspect_labels and crop_aspect_ratios must have the same length")
        out: List[tuple[str, tuple[float, float]]] = []
        for label, ratio in zip(labels, ratios):
            if len(ratio) != 2:
                raise ValueError(f"Each crop aspect ratio must be a pair (w, h), got: {ratio!r}")
            aw = float(ratio[0])
            ah = float(ratio[1])
            if aw <= 0 or ah <= 0:
                raise ValueError(f"Crop aspect values must be > 0, got: {ratio!r}")
            out.append((str(label), (aw, ah)))
        return out

    def run(self) -> None:
        DatasetSchema.init_csv(self.cfg.csv_path)

        N = int(self.cfg.num_slices)
        aspect_specs = self._iter_crop_aspects()
        total_rows = N * (1 + 3 * len(aspect_specs))
        planes: List[Tuple[Vec3, float, float]] = []

        pbar = tqdm(
            total=total_rows,
            desc="Generating dataset (full + multi-aspect crops)",
            dynamic_ncols=True,
            leave=True,
        )

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
                    planes.append((n_vec, float(depth), float(rotation)))
                    self._write_full(plane_id, a_slice, r_slice, n_vec, depth, rotation)
                    self.stats.full_written += 1
                    pbar.update(1)
                    pbar.set_postfix(self.stats.postfix())
                    break
                self.stats.planes_resampled += 1
                pbar.set_postfix(self.stats.postfix())

        for plane_id in range(N):
            n_vec, depth, rotation = planes[plane_id]
            a_slice, r_slice = self.volumes.sample_pair(
                normal=n_vec,
                depth=depth,
                rotation=rotation,
                size=self.cfg.slice_size,
            )

            for crop_idx in (0, 1, 2):
                for crop_kind, (aspect_w, aspect_h) in aspect_specs:
                    self._write_one_crop_for_plane(
                        plane_id=plane_id,
                        crop_idx=crop_idx,
                        crop_kind=crop_kind,
                        aspect_w=aspect_w,
                        aspect_h=aspect_h,
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
            crop_bin="full",
            crop_kind="full",
            crop_aspect_w=1.0,
            crop_aspect_h=1.0,
        )
        DatasetSchema.append_row(self.cfg.csv_path, row)

    def _write_one_crop_for_plane(
        self,
        plane_id: int,
        crop_idx: int,
        crop_kind: str,
        aspect_w: float,
        aspect_h: float,
        a_slice: Slice,
        r_slice: Slice,
        n_vec: Vec3,
        depth: float,
        rotation: float,
    ) -> None:
        max_attempts = int(self.cfg.max_crop_attempts)
        if max_attempts <= 0:
            raise ValueError("max_crop_attempts must be > 0 to generate fixed crop quotas")

        crop_bin_label = self.CROP_BIN_LABELS[int(crop_idx)]

        for _ in range(max_attempts):
            self.stats.crop_attempts += 1
            cx, cy, rw, rh = self.crop_sampler.sample_crop_params_for_bin_and_aspect(
                crop_idx,
                aspect_w=float(aspect_w),
                aspect_h=float(aspect_h),
            )

            a_crop = a_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)
            r_crop = r_slice.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)

            if not self.volumes.allen.is_valid_slice(a_crop):
                continue

            suffix = f"crop{crop_idx}_{crop_kind}"
            allen_path = self.cfg.out_dir / f"{plane_id:05d}_a_{suffix}.png"
            real_path = self.cfg.out_dir / f"{plane_id:05d}_r_{suffix}.png"

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
                crop_bin=crop_bin_label,
                crop_kind=str(crop_kind),
                crop_aspect_w=float(aspect_w),
                crop_aspect_h=float(aspect_h),
            )
            DatasetSchema.append_row(self.cfg.csv_path, row)
            return

        self.stats.crop_failures += 1
        raise RuntimeError(
            f"Failed to generate a valid crop for plane_id={plane_id}, crop_idx={crop_idx}, "
            f"crop_kind={crop_kind} within max_crop_attempts={max_attempts}. "
            "Consider increasing --max-crop-attempts or relaxing --min-crop-frac/--max-crop-frac."
        )
