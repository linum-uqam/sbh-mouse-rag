# scripts/run_eval.py
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import faiss

from eval.evaluator import Evaluator
from eval.config import EvalConfig
from dataset import DatasetConfig  # for default CSV path and volume defaults


SEARCH_MODE_TO_LOCAL = {
    "fast": "off",
    "smart": "auto",
    "enhanced": "force",
}


def _parse_force_square_scales(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return (2,)
    txt = str(raw).strip()
    if not txt:
        return tuple()
    vals = []
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return tuple(vals)


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
    p.add_argument("--no-crop", action="store_true", help="Disable auto foreground crop on query.")
    p.add_argument("--no-flip-x", action="store_true", help="Disable horizontal flip augmentation.")
    p.add_argument("--no-flip-y", action="store_true", help="Disable vertical flip augmentation.")
    p.add_argument("--no-pad-to-square", action="store_true", help="Disable square padding before embedding.")
    p.add_argument("--debug", action="store_true", default=False, help="Verbose logging.")

    # High-level search mode control
    p.add_argument(
        "--search-mode",
        type=str,
        default="fast",
        choices=["fast", "smart", "enhanced", "all"],
        help="Search mode to evaluate. 'all' runs fast, smart, and enhanced sequentially.",
    )

    # Fine-grained local query expansion controls
    p.add_argument("--local-k-per-view", type=int, default=None, help="Neighbors fetched per local crop view (default: k-per-angle).")
    p.add_argument("--local-score-mode", type=str, default="top2_mean", choices=["max", "top2_mean"], help="How to aggregate local crop evidence.")
    p.add_argument("--global-weight", type=float, default=1.0, help="Weight for global-query matches.")
    p.add_argument("--local-weight", type=float, default=0.35, help="Weight for local-crop matches.")
    p.add_argument("--auto-local-aspect-threshold", type=float, default=1.35, help="Aspect-ratio threshold that activates smart mode local crops.")
    p.add_argument("--local-crop-overlap", type=float, default=0.50, help="Overlap ratio between local square crops.")
    p.add_argument("--local-crop-min-side-px", type=int, default=64, help="Minimum side length for local crops.")
    p.add_argument("--auto-max-local-crops", type=int, default=3, help="Max local crops in smart mode.")
    p.add_argument("--force-max-local-crops", type=int, default=8, help="Max local crops in enhanced mode.")
    p.add_argument("--force-square-scales", type=str, default="2", help="Comma-separated extra square crop scales for enhanced mode, e.g. '2' or '2,3'.")

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

    # -------------------- Resume / overwrite / memory --------------------
    p.add_argument("--overwrite", action="store_true", help="Delete/recreate eval_hits.csv before starting.")
    p.add_argument("--max-retrieved-slice-cache", type=int, default=256, help="Max number of reconstructed Allen plane slices kept in RAM.")
    p.add_argument("--csv-flush-every", type=int, default=128, help="Flush CSV every N written rows.")
    p.add_argument("--gc-every-rows", type=int, default=25, help="Run gc.collect() every N processed dataset rows.")

    return p.parse_args()


def _build_cfg(a: argparse.Namespace, mode_name: str) -> EvalConfig:
    rerank_topk = a.rerank_topk if a.rerank_topk is not None else a.final_k
    base_save_dir = Path(a.save_dir) if a.save_dir else EvalConfig.save_dir
    save_dir = base_save_dir / mode_name if a.search_mode == "all" else base_save_dir

    return EvalConfig(
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
        flip_x=not a.no_flip_x,
        flip_y=not a.no_flip_y,
        pad_to_square=not a.no_pad_to_square,
        debug=a.debug,
        search_mode_label=mode_name,
        local_search_mode=SEARCH_MODE_TO_LOCAL[mode_name],
        local_k_per_view=a.local_k_per_view,
        local_score_mode=a.local_score_mode,
        local_weight=a.local_weight,
        global_weight=a.global_weight,
        auto_local_aspect_threshold=a.auto_local_aspect_threshold,
        local_crop_overlap=a.local_crop_overlap,
        local_crop_min_side_px=a.local_crop_min_side_px,
        auto_max_local_crops=a.auto_max_local_crops,
        force_max_local_crops=a.force_max_local_crops,
        force_square_scales=_parse_force_square_scales(a.force_square_scales),

        # distance
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
        save_dir=save_dir,
        save_k=a.save_k,
        save_seed=a.save_seed,

        # resume / overwrite / memory
        overwrite=a.overwrite,
        max_retrieved_slice_cache=a.max_retrieved_slice_cache,
        csv_flush_every=a.csv_flush_every,
        gc_every_rows=a.gc_every_rows,
    )


def main() -> None:
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    a = parse_args()

    modes = ["fast", "smart", "enhanced"] if a.search_mode == "all" else [a.search_mode]

    for i, mode_name in enumerate(modes):
        print(f"\n=== Running evaluation mode: {mode_name} ===")
        cfg = _build_cfg(a, mode_name)

        # Prevent accidental carry-over deletions when looping through all modes.
        if a.search_mode == "all" and i > 0:
            cfg.overwrite = a.overwrite

        Evaluator(cfg).run()


if __name__ == "__main__":
    main()
