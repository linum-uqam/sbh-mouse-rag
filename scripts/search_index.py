from __future__ import annotations

import argparse
from pathlib import Path

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig
from index.utils import log, load_image_gray
from index.config import OUT_DIR
from index.vis import save_hits_only_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the patch index with a query image."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to query image (2D mouse brain slice or crop).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top results to return (after merging variants).",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 180.0, 270.0],
        help="Query rotation angles in degrees (default: 0 90 180 270).",
    )
    parser.add_argument(
        "--k-per-angle",
        type=int,
        default=64,
        help="Number of neighbours to fetch per global query variant (default: 64).",
    )
    parser.add_argument(
        "--local-k-per-view",
        type=int,
        default=None,
        help="Number of neighbours to fetch per local crop variant. Default: same as --k-per-angle.",
    )

    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional indexed patch scales to search in. "
            "Examples: --scales 1 (whole slice only), --scales 2, --scales 4, "
            "--scales 1 2. Default: all scales."
        ),
    )

    # Default = enabled
    parser.add_argument(
        "--no-flip-x",
        action="store_true",
        help="Disable horizontal flip augmentation on the query image.",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Disable vertical flip augmentation on the query image.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable auto foreground cropping on the query image.",
    )
    parser.add_argument(
        "--no-pad-to-square",
        action="store_true",
        help="Disable square padding before DINO preprocessing.",
    )

    parser.add_argument(
        "--local-search-mode",
        choices=["off", "auto", "force"],
        default="auto",
        help="Local crop search mode: off, auto, or force (default: auto).",
    )
    parser.add_argument(
        "--local-score-mode",
        choices=["max", "top2_mean"],
        default="top2_mean",
        help="How to aggregate local crop scores per candidate (default: top2_mean).",
    )
    parser.add_argument(
        "--global-weight",
        type=float,
        default=1.0,
        help="Weight for global branch scores (default: 1.0).",
    )
    parser.add_argument(
        "--local-weight",
        type=float,
        default=0.35,
        help="Weight for local crop branch scores (default: 0.35).",
    )
    parser.add_argument(
        "--auto-local-aspect-threshold",
        type=float,
        default=1.35,
        help="Aspect-ratio threshold that automatically enables local crop search in auto mode (default: 1.35).",
    )
    parser.add_argument(
        "--local-crop-overlap",
        type=float,
        default=0.50,
        help="Overlap ratio between neighbouring local square crops (default: 0.50).",
    )
    parser.add_argument(
        "--local-crop-min-side-px",
        type=int,
        default=64,
        help="Minimum allowed side length for a local crop (default: 64).",
    )
    parser.add_argument(
        "--auto-max-local-crops",
        type=int,
        default=3,
        help="Maximum number of local crops in auto mode (default: 3).",
    )
    parser.add_argument(
        "--force-max-local-crops",
        type=int,
        default=8,
        help="Maximum number of local crops in force mode (default: 8).",
    )
    parser.add_argument(
        "--force-square-scales",
        type=int,
        nargs="+",
        default=[2],
        help="Extra square crop grid scales used only in force mode (default: 2).",
    )

    parser.add_argument(
        "--index-root",
        type=Path,
        default=OUT_DIR,
        help=f"Directory containing patch_index.faiss & patch_manifest.parquet (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save visualizations of the top-k hits into this directory.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="If set, apply neural reranker on the top-k results.",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=Path("out/reranker/reranker.pt"),
        help="Path to trained reranker model (default: out/reranker/reranker.pt).",
    )
    parser.add_argument(
        "--rerank-topk",
        type=int,
        default=10,
        help="How many of the coarse hits to rerank (default: 10).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    flip_x = not args.no_flip_x
    flip_y = not args.no_flip_y
    global_variant_count = len(args.angles) * (2 if flip_x else 1) * (2 if flip_y else 1)
    allowed_scales = None if args.scales is None else tuple(sorted({int(s) for s in args.scales}))

    log("search", [
        "Starting search...",
        f"Image                      : {args.image}",
        f"k                          : {args.k}",
        f"Angles (deg)               : {args.angles}",
        f"k_per_angle                : {args.k_per_angle}",
        f"local_k_per_view           : {args.local_k_per_view}",
        f"Allowed scales             : {'all' if allowed_scales is None else allowed_scales}",
        f"flip_x                     : {flip_x}",
        f"flip_y                     : {flip_y}",
        f"Global query variants      : {global_variant_count}",
        f"Auto-crop                  : {not args.no_crop}",
        f"Pad to square              : {not args.no_pad_to_square}",
        f"Local search mode          : {args.local_search_mode}",
        f"Local score mode           : {args.local_score_mode}",
        f"Global weight              : {args.global_weight}",
        f"Local weight               : {args.local_weight}",
        f"Auto local aspect thr      : {args.auto_local_aspect_threshold}",
        f"Local crop overlap         : {args.local_crop_overlap}",
        f"Local crop min side px     : {args.local_crop_min_side_px}",
        f"Auto max local crops       : {args.auto_max_local_crops}",
        f"Force max local crops      : {args.force_max_local_crops}",
        f"Force square scales        : {args.force_square_scales}",
        f"Index root                 : {args.index_root}",
        f"Save dir                   : {args.save_dir}",
        f"Use reranker               : {args.use_reranker}",
    ])

    store = IndexStore(root=args.index_root).load_all()

    cfg = SearchConfig(
        angles=tuple(float(a) for a in args.angles),
        allowed_scales=allowed_scales,
        flip_x=flip_x,
        flip_y=flip_y,
        pad_to_square=not args.no_pad_to_square,
        k_per_angle=int(args.k_per_angle),
        local_k_per_view=(None if args.local_k_per_view is None else int(args.local_k_per_view)),
        crop_foreground=not args.no_crop,
        local_search_mode=str(args.local_search_mode),
        local_score_mode=str(args.local_score_mode),
        global_weight=float(args.global_weight),
        local_weight=float(args.local_weight),
        auto_local_aspect_threshold=float(args.auto_local_aspect_threshold),
        local_crop_overlap=float(args.local_crop_overlap),
        local_crop_min_side_px=int(args.local_crop_min_side_px),
        auto_max_local_crops=int(args.auto_max_local_crops),
        force_max_local_crops=int(args.force_max_local_crops),
        force_square_scales=tuple(int(x) for x in args.force_square_scales),
        use_reranker=args.use_reranker,
        rerank_topk=args.rerank_topk,
        reranker_model_path=args.reranker_model,
        reranker_device="cuda",
        reranker_batch_size=32,
    )

    searcher = SliceSearcher(store, cfg=cfg)
    img = load_image_gray(args.image)
    hits, query_img = searcher.search_image(img, k=args.k)

    df = searcher.to_dataframe(hits)

    log("search", [f"Top {len(hits)} results:"])
    for i, h in enumerate(hits, start=1):
        m = h.meta
        normal_idx = m.get("normal_idx", "?")
        depth_idx = m.get("depth_idx", "?")
        scale = m.get("scale", "?")
        x0, y0, x1, y1 = (
            m.get("x0", "?"),
            m.get("y0", "?"),
            m.get("x1", "?"),
            m.get("y1", "?"),
        )
        depth_vox = m.get("depth_vox", "?")
        rotation_deg = m.get("rotation_deg", "?")
        extra = f" rerank={h.rerank_score:.4f}" if h.rerank_score is not None else ""

        log("", [
            f"[{i:02d}] score={h.score:.4f}{extra}",
            f"(query_angle={h.angle:.1f}°, flip_x={h.flip_x}, flip_y={h.flip_y}, patch_id={h.patch_id}, branch={h.query_branch}, crop_id={h.query_crop_id})",
            f"normal={normal_idx} depth={depth_idx} scale={scale}",
            f"box=({x0},{y0})-({x1},{y1}) depth_vox={depth_vox:.3f} rot_deg={rotation_deg:.1f}\n",
        ])

    if args.save_dir is not None:
        save_hits_only_images(
            hits,
            out_dir=Path(args.save_dir),
            mode="patch",
            top_n=len(hits),
            verbose=True,
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / "search_results.csv"
        df.to_csv(csv_path, index=False)
        log("search", [f"Saved CSV results to: {csv_path}"])


if __name__ == "__main__":
    main()
