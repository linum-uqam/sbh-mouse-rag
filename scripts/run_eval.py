# scripts/run_eval.py
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import faiss

from eval.evaluator import Evaluator
from eval.config import EvalConfig
from dataset import DatasetConfig  # for default CSV path and volume defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dataset-driven eval loop that writes eval_hits.csv with geometry distances + listwise soft targets."
    )

    default_ds = DatasetConfig()
    default_csv = str(default_ds.csv_path)

    # -------------------- Data --------------------
    p.add_argument("--csv", type=str, default=default_csv, help=f"Dataset CSV (default: {default_csv})")
    p.add_argument("--source", type=str, default="allen", choices=["allen", "real", "both"], help="Which modality to evaluate.")
    p.add_argument("--limit", type=int, default=None, help="Max number of rows to evaluate (None = all).")

    # -------------------- Volumes --------------------
    p.add_argument("--allen-cache-dir", type=str, default=default_ds.allen_cache_dir, help="Allen cache dir.")
    p.add_argument("--allen-res-um", type=int, default=default_ds.allen_res_um, help="Allen resolution in microns (metadata).")
    p.add_argument("--real-volume-path", type=str, default=str(default_ds.real_volume_path), help="Real NIfTI path (only used if source includes real).")

    # -------------------- Slice sampling (query slices) --------------------
    p.add_argument("--size-px", type=int, default=512, help="Slice size (square) when sampling queries.")
    p.add_argument("--pixel-step-vox", type=float, default=1.0, help="Voxel step per pixel for slicing.")
    p.add_argument("--no-linear-interp", action="store_true", help="Disable linear interpolation (use nearest).")

    # -------------------- Search --------------------
    p.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270], help="Query rotation angles (deg).")
    p.add_argument("--final-k", type=int, default=100, help="Top-K hits to keep per query.")
    p.add_argument("--k-per-angle", type=int, default=64, help="Neighbors fetched per rotation angle.")
    p.add_argument("--no-crop", action="store_true", help="Disable auto foreground crop on query image.")
    p.add_argument("--debug", action="store_true", default=False, help="Verbose logging.")

    # -------------------- Geometry distance (Slice.distance) --------------------
    p.add_argument("--distance-grid", type=int, default=64, help="Grid size for Slice.distance (grid x grid points).")
    p.add_argument("--distance-trim", type=float, default=0.10, help="Trim fraction for robust distance (0 disables).")

    # -------------------- Optional reranker passthrough --------------------
    p.add_argument("--use-reranker", action="store_true", default=False, help="Enable neural reranker on top of coarse search.")
    p.add_argument("--rerank-topk", type=int, default=None, help="Only rerank top-K coarse hits (default: final-k).")
    p.add_argument("--reranker-model-path", type=str, default="out/reranker/reranker.pt", help="Path to reranker .pt file.")
    p.add_argument("--reranker-device", type=str, default="cuda", help='Device for reranker inference ("cuda" or "cpu").')
    p.add_argument("--reranker-batch-size", type=int, default=32, help="Batch size for reranker scoring.")

    # -------------------- Output --------------------
    p.add_argument("--save-dir", type=str, default=None, help="Output directory (default: EvalConfig.save_dir).")
    p.add_argument("--save-k", type=int, default=None, help="Save exactly K random rows of visuals (optional).")
    p.add_argument("--save-seed", type=int, default=123, help="Seed for selecting which rows to visualize.")

    return p.parse_args()


def main() -> None:
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    a = parse_args()

    rerank_topk = a.rerank_topk if a.rerank_topk is not None else a.final_k

    cfg = EvalConfig(
        # data
        csv_path=Path(a.csv),
        source=a.source,
        limit=a.limit,

        # volumes
        allen_cache_dir=a.allen_cache_dir,
        allen_res_um=a.allen_res_um,
        real_volume_path=Path(a.real_volume_path) if a.real_volume_path else None,

        # slicing
        size_px=a.size_px,
        pixel_step_vox=a.pixel_step_vox,
        linear_interp=not a.no_linear_interp,

        # search
        angles=tuple(a.angles),
        final_k=a.final_k,
        k_per_angle=a.k_per_angle,
        crop_foreground=not a.no_crop,
        debug=a.debug,

        # distance (voxel-space only)
        distance_grid=a.distance_grid,
        distance_trim_frac=a.distance_trim,
        distance_physical=False,

        # reranker passthrough
        use_reranker=a.use_reranker,
        rerank_topk=rerank_topk,
        reranker_model_path=Path(a.reranker_model_path),
        reranker_device=a.reranker_device,
        reranker_batch_size=a.reranker_batch_size,

        # output
        save_dir=Path(a.save_dir) if a.save_dir else EvalConfig.save_dir,
        save_k=a.save_k,
        save_seed=a.save_seed,
    )

    Evaluator(cfg).run()


if __name__ == "__main__":
    main()
