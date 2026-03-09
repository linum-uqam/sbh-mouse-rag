# scripts/create_index.py
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import faiss

from volume.volume_helper import AllenVolume
from index.patch_index import PatchIndexBuilder, PatchSamplingConfig, IndexConfig
from index.config import (
    OUT_DIR,
    K_NORMALS,
    SLICE_SIZE,
    PATCH_SCALES,
    PATCH_OVERLAP,
    INDEX_STRATEGY,
    D,
    FIXED_STEP_VOX,
    FIXED_MARGIN_VOX,
    FIXED_ROTATIONS,
)
from index.utils import log


def _parse_rotations(vals: Sequence[float]) -> tuple[float, ...]:
    rots = tuple(float(v) for v in vals)
    # Keep pipeline valid: at least one rotation
    return rots if len(rots) > 0 else (0.0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-scale patch index over the Allen average mouse brain."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory for index + manifest (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--k-normals",
        type=int,
        default=K_NORMALS,
        help=f"Number of Fibonacci normals (default: {K_NORMALS})",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=SLICE_SIZE,
        help=f"Slice size in pixels (square) (default: {SLICE_SIZE})",
    )
    parser.add_argument(
        "--patch-scales",
        type=int,
        nargs="+",
        default=list(PATCH_SCALES),
        help=f"Patch scales (base tiles per side) (default: {PATCH_SCALES})",
    )
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=PATCH_OVERLAP,
        help=f"Patch overlap ratio in (0,1), e.g. 0.5 for 50%% (default: {PATCH_OVERLAP})",
    )

    # depth/rotation controls (slice sampling)
    parser.add_argument(
        "--depth-step-vox",
        type=float,
        default=float(FIXED_STEP_VOX),
        help=f"Depth sampling step along normal, in voxels (default: {FIXED_STEP_VOX})",
    )
    parser.add_argument(
        "--depth-margin-vox",
        type=float,
        default=float(FIXED_MARGIN_VOX),
        help=f"Margin removed from depth bounds, in voxels (default: {FIXED_MARGIN_VOX})",
    )
    parser.add_argument(
        "--rotations-deg",
        type=float,
        nargs="+",
        default=list(FIXED_ROTATIONS),
        help=f"In-plane rotations (degrees) applied per (normal, depth) (default: {FIXED_ROTATIONS})",
    )

    parser.add_argument(
        "--index-strategy",
        type=str,
        choices=["auto", "flat", "hnsw", "ivfpq"],
        default=INDEX_STRATEGY,
        help=f"Index strategy (default: {INDEX_STRATEGY})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Basic validation
    if not (0.0 < float(args.patch_overlap) < 1.0):
        raise ValueError("--patch-overlap must be in (0,1)")
    if float(args.depth_step_vox) <= 0.0:
        raise ValueError("--depth-step-vox must be > 0")
    if float(args.depth_margin_vox) < 0.0:
        raise ValueError("--depth-margin-vox must be >= 0")

    rotations_deg = _parse_rotations(args.rotations_deg)

    # Use most CPU cores but leave 1 free.
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))

    log("main", [
        "Starting patch index creation...",
        f"OUT_DIR          : {args.out_dir}",
        f"K_NORMALS        : {args.k_normals}",
        f"SLICE_SIZE       : {args.slice_size}",
        f"PATCH_SCALES     : {tuple(args.patch_scales)}",
        f"PATCH_OVERLAP    : {args.patch_overlap}",
        f"DEPTH_STEP_VOX   : {args.depth_step_vox}",
        f"DEPTH_MARGIN_VOX : {args.depth_margin_vox}",
        f"ROTATIONS_DEG    : {rotations_deg} (n={len(rotations_deg)})",
        f"INDEX_STRATEGY   : {args.index_strategy}",
    ])

    # 1) Load Allen volume
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)

    # 2) Build configs
    sampling_cfg = PatchSamplingConfig(
        depth_step_vox=float(args.depth_step_vox),
        depth_margin_vox=float(args.depth_margin_vox),
        rotations_deg=tuple(rotations_deg),
        slice_size_px=int(args.slice_size),
        patch_scales=tuple(int(s) for s in args.patch_scales),
        patch_overlap=float(args.patch_overlap),
    )

    index_cfg = IndexConfig(
        dim=int(D),
        strategy=str(args.index_strategy),
    )

    # 3) Build & save index
    builder = PatchIndexBuilder(
        vol_helper=allen,
        k_normals=int(args.k_normals),
        sampling_cfg=sampling_cfg,
        index_cfg=index_cfg,
    )

    index_path, manifest_path = builder.run(args.out_dir)

    log("main", [
        "Patch index creation complete.",
        f"Index saved    : {index_path}",
        f"Manifest saved : {manifest_path}",
    ])


if __name__ == "__main__":
    main()