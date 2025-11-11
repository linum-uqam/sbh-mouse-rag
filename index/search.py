from __future__ import annotations
import argparse
from pathlib import Path
import multiprocessing as mp
import faiss

from index.store import IndexStore
from index.searcher import SliceSearcher
from index.utils import load_image
from index.visual import save_results_images

def main():
    p = argparse.ArgumentParser(description="2D→3D search with hybrid coarse retrieval + optional ColBERT rerank")
    p.add_argument("image", type=str, help="Path to query image (png/jpg/tif)")

    p.add_argument("--mode", type=str, default="col",
                   choices=["base", "col"],
                   help="base: hybrid Stage-1 only. col: add ColBERT-style token rerank.")

    p.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 14],
                   help="Query token grid sizes for angle×scale embedding.")
    p.add_argument("--final-k", type=int, default=10)

    # Stage-1 knobs
    p.add_argument("--base-per-rotation", type=int, default=200,
                   help="Per-rotation K for global & per-scale coarse indices.")
    p.add_argument("--base-topk-for-col", type=int, default=300,
                   help="Total candidate budget merged for Stage-2 rerank.")

    # Token-ANN knobs
    p.add_argument("--use-token-ann", action="store_true", default=True,
                   help="Enable token IVF-PQ coarse shortlist (recommended for crop queries).")
    p.add_argument("--token-scales", type=int, nargs="+", default=[4, 8, 14],
                   help="Which query scales to use for token ANN.")
    p.add_argument("--token-topM", type=int, default=32,
                   help="TopM per query token in token ANN search.")
    
    p.add_argument("--debug", action="store_true", help="Print Stage-1 diagnostics and candidate counts.")

    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, dump top-K result images for manual QA.")
    args = p.parse_args()

    store = IndexStore().load_all()
    searcher = SliceSearcher(store)

    img = load_image(Path(args.image))
    hits = searcher.search_image(
        img_np=img,
        angles=args.angles,
        scales=args.scales,
        mode=args.mode,
        base_topk_for_col=args.base_topk_for_col,
        base_per_rotation=args.base_per_rotation,
        token_topM=args.token_topM,
        token_scales=args.token_scales,
        use_token_ann=args.use_token_ann,
        rerank_final_k=args.final_k,
        debug=args.debug,   
    )

    if not hits:
        print("No results.")
        return

    for i, h in enumerate(hits, 1):
        h["rank"] = i

    print("\nTop results:")
    for h in hits:
        n = h["normal"]
        cols = [
            f"{h['rank']:02d}. id={h['slice_id']}",
            f"score={h['score']:.4f}",
            f"best_rot={h['best_angle']:>6.1f}°",
            f"base={h['base_score']:.4f}",
        ]
        if "col_score" in h:
            cols.append(f"col={h['col_score']:.4f}")
        if "best_scale" in h:
            cols.append(f"scale={h['best_scale']}x{h['best_scale']}")
        cols.extend([f"n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})", f"depth={h['depth_vox']:.1f}"])
        print("  " + "  ".join(cols))

    if args.save_dir:
        save_results_images(hits, Path(args.save_dir))
        print(f"\nSaved result images -> {args.save_dir}")

if __name__ == "__main__":
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    main()
