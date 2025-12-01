# scripts/create_index.py
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

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
)
from index.utils import log


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

    # Use most CPU cores but leave 1 free.
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))

    log("main", [
        "Starting patch index creation...",
        f"OUT_DIR        : {args.out_dir}",
        f"K_NORMALS      : {args.k_normals}",
        f"SLICE_SIZE     : {args.slice_size}",
        f"PATCH_SCALES   : {tuple(args.patch_scales)}",
        f"PATCH_OVERLAP  : {args.patch_overlap}",
        f"INDEX_STRATEGY : {args.index_strategy}",
    ])

    # 1) Load Allen volume
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)

    # 2) Build configs
    sampling_cfg = PatchSamplingConfig(
        slice_size_px=args.slice_size,
        patch_scales=tuple(args.patch_scales),
        patch_overlap=args.patch_overlap,
    )

    index_cfg = IndexConfig(
        dim=D,
        strategy=args.index_strategy,
    )

    # 3) Build & save index
    builder = PatchIndexBuilder(
        vol_helper=allen,
        k_normals=args.k_normals,
        sampling_cfg=sampling_cfg,
        index_cfg=index_cfg,
    )

    index_path, manifest_path = builder.run(args.out_dir)

    log("main", [
        "Patch index creation complete.",
        f"Index saved   : {index_path}",
        f"Manifest saved: {manifest_path}",
    ])


if __name__ == "__main__":
    main()
