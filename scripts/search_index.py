from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig, SearchResult
from index.utils import log, load_image_gray
from index.config import OUT_DIR
from index.vis import save_search_results_visuals, save_hits_only_images


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
        help="Number of top results to return (after merging rotations/flips).",
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
        help="Number of neighbours to fetch per query variant (default: 64).",
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
    query_variant_count = len(args.angles) * (2 if flip_x else 1) * (2 if flip_y else 1)

    log("search", [
        "Starting search...",
        f"Image         : {args.image}",
        f"k             : {args.k}",
        f"Angles (deg)  : {args.angles}",
        f"k_per_angle   : {args.k_per_angle}",
        f"flip_x        : {flip_x}",
        f"flip_y        : {flip_y}",
        f"Query variants: {query_variant_count}",
        f"Auto-crop     : {not args.no_crop}",
        f"Index root    : {args.index_root}",
        f"Save dir      : {args.save_dir}",
        f"Use reranker  : {args.use_reranker}",
    ])

    # 1) Load store (index + manifest)
    store = IndexStore(root=args.index_root).load_all()

    # 2) Build search config
    cfg = SearchConfig(
        angles=tuple(float(a) for a in args.angles),
        flip_x=flip_x,
        flip_y=flip_y,
        k_per_angle=int(args.k_per_angle),
        crop_foreground=not args.no_crop,
        use_reranker=args.use_reranker,
        rerank_topk=args.rerank_topk,
        reranker_model_path=args.reranker_model,
        reranker_device="cuda",       # or make this a CLI arg if you want
        reranker_batch_size=32,
    )

    # 3) Create searcher and run (getting hits + preprocessed query image)
    searcher = SliceSearcher(store, cfg=cfg)
    img = load_image_gray(args.image)
    hits, query_img = searcher.search_image(img, k=args.k)

    df = searcher.to_dataframe(hits)

    # Pretty print basic info for each hit
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
            f"[{i:02d}] score={h.score:.4f}{extra} ",
            f"(query_angle={h.angle:.1f}°, flip_x={h.flip_x}, flip_y={h.flip_y}, patch_id={h.patch_id}) ",
            f"normal={normal_idx} depth={depth_idx} scale={scale} ",
            f"box=({x0},{y0})-({x1},{y1}) depth_vox={depth_vox:.3f} rot_deg={rotation_deg:.1f}\n"
        ])

    # 4) Optionally save visuals
    if args.save_dir is not None:
        # save_search_results_visuals(hits, query_img, args.save_dir)
        save_hits_only_images(
            hits,
            out_dir=Path(args.save_dir),
            mode="patch",   # <-- change to "full" if you want full slices
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