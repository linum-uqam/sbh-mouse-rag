# index/search.py
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
    p = argparse.ArgumentParser(description="2D→3D search with optional ColBERT-style rerank")
    p.add_argument("image", type=str, help="Path to query image (png/jpg/tif)")
    p.add_argument("--mode", type=str, default="base",
                   choices=["base", "col"],
                   help="base: global embeddings only. col: ColBERT-style token rerank.")
    p.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 14],
               help="Query token grid sizes for multi-scale ColBERT (adaptive pooled).")
    p.add_argument("--base-topk-for-col", type=int, default=200,
                   help="How many base hits to pass to the col reranker.")
    p.add_argument("--base-per-rotation", type=int, default=200,
                   help="Base search candidates per rotation.")
    p.add_argument("--final-k", type=int, default=10)
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
        rerank_final_k=args.final_k,
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
