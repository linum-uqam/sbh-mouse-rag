# scripts/run_eval.py
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import faiss

from eval.evaluator import Evaluator
from dataset import DatasetConfig  # for default CSV path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dataset-driven eval loop over FAISS+DINOv3 patch index"
    )

    # Data
    default_cfg = DatasetConfig()  # uses same defaults as builder
    default_csv = str(default_cfg.csv_path)

    p.add_argument(
        "--csv",
        type=str,
        default=default_csv,
        help=f"Path to dataset CSV (default: {default_csv})",
    )
    p.add_argument(
        "--source",
        type=str,
        default="allen",
        choices=["allen", "real", "both"],
        help="Which modality to evaluate.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to evaluate (None = all).",
    )
    p.add_argument(
        "--include-annotation",
        action="store_true",
        default=True,
        help="Sample Allen annotations for slices when available.",
    )

    # Volumes
    p.add_argument(
        "--allen-cache-dir",
        type=str,
        default=default_cfg.allen_cache_dir,
        help=f"Allen cache dir (default: {default_cfg.allen_cache_dir}).",
    )
    p.add_argument(
        "--allen-res-um",
        type=int,
        default=default_cfg.allen_res_um,
        help=f"Allen resolution in microns (default: {default_cfg.allen_res_um}).",
    )
    p.add_argument(
        "--real-volume-path",
        type=str,
        default=str(default_cfg.real_volume_path),
        help=f"Real NIfTI volume path (default: {default_cfg.real_volume_path}).",
    )

    # Slice sampling (dataset-side sampling at eval time)
    p.add_argument(
        "--size-px",
        type=int,
        default=512,
        help="Slice size (square) when sampling from volumes at eval time.",
    )
    p.add_argument(
        "--pixel-step-vox",
        type=float,
        default=1.0,
        help="Voxel step per pixel for eval-time slicing.",
    )
    p.add_argument(
        "--linear-interp",
        action="store_true",
        default=True,
        help="Use linear interpolation in sampling (nearest if False).",
    )

    # Search
    p.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[0, 90, 180, 270],
        help="Query rotation angles in degrees.",
    )
    p.add_argument(
        "--final-k",
        type=int,
        default=10,
        help="Top-k patches to keep per query.",
    )
    p.add_argument(
        "--k-per-angle",
        type=int,
        default=64,
        help="Neighbours fetched per rotation.",
    )
    p.add_argument(
        "--no-crop",
        action="store_true",
        default=False,
        help="Disable auto foreground cropping on the query image.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Verbose logging of per-row hits.",
    )

    # Output
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visual examples and eval CSV.",
    )
    p.add_argument(
        "--save-k",
        type=int,
        default=None,
        help="Save exactly K random rows among processed ones (if save-dir is set).",
    )
    p.add_argument(
        "--save-seed",
        type=int,
        default=123,
        help="Seed for selecting which rows to visualize.",
    )

    return p.parse_args()


def main() -> None:
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    a = parse_args()

    Evaluator(
        csv_path=a.csv,
        source=a.source,
        limit=a.limit,
        include_annotation=a.include_annotation,
        allen_cache_dir=a.allen_cache_dir,
        allen_res_um=a.allen_res_um,
        real_volume_path=a.real_volume_path,
        size_px=a.size_px,
        pixel_step_vox=a.pixel_step_vox,
        linear_interp=a.linear_interp,
        angles=tuple(a.angles),
        final_k=a.final_k,
        k_per_angle=a.k_per_angle,
        crop_foreground=not a.no_crop,
        debug=a.debug,
        save_dir=a.save_dir,
        save_k=a.save_k,
        save_seed=a.save_seed,
    ).run()


if __name__ == "__main__":
    main()
