# index/scripts/search_top1_vs_rot180.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# add this import at the top
from volume.volume_helper import AllenVolume

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig
from index.utils import load_image_gray, log
from index.config import OUT_DIR

from index.model.dino import model as dino


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Compare top1 retrieval vs rotated-query cosine.")
    ap.add_argument("image", type=Path, help="Query image path.")
    ap.add_argument("--index-root", type=Path, default=OUT_DIR, help="Index root (default: OUT_DIR).")
    ap.add_argument("--out", type=Path, default=Path("out/top1_vs_rot180.png"), help="Output figure path.")
    ap.add_argument("--no-crop", action="store_true", help="Disable auto foreground cropping on the query.")
    return ap.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # your dino wrapper already L2-normalizes, dot product is cosine; keep safe anyway
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def save_thesis_figure(
    *,
    query_img: np.ndarray,
    top1_img: np.ndarray,
    query_rot: np.ndarray,
    sim_q_top1: float,
    sim_q_rot: float,
    out_path: Path,
    variant: str = "fr",  # "fr" or "math"
) -> None:
    import matplotlib.pyplot as plt

    # Thesis-ish defaults (no LaTeX dependency)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "cm",
    })

    delta = sim_q_top1 - sim_q_rot

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(query_img, cmap="gray")
    axes[1].imshow(top1_img, cmap="gray")
    axes[2].imshow(query_rot, cmap="gray")

    if variant == "fr":
        axes[0].set_title("Requête $q$")
        axes[1].set_title(
            "Top-1 $x_1$\n" + rf"$\cos(q,x_1)={sim_q_top1:.4f}$"
        )
        axes[2].set_title(
            r"Rotation $180^\circ$" + "\n" + rf"$\cos(q,\mathrm{{rot}}_{{180}}(q))={sim_q_rot:.4f}$"
        )
        footer = rf"$\Delta=\cos(q,x_1)-\cos(q,\mathrm{{rot}}_{{180}}(q))={delta:.4f}$"
    else:
        # math-only / language-neutral
        axes[0].set_title(r"(a) $q$")
        axes[1].set_title(rf"(b) $x_1$  |  $\cos(q,x_1)={sim_q_top1:.4f}$")
        axes[2].set_title(rf"(c) $\mathrm{{rot}}_{{180}}(q)$  |  $\cos(q,\mathrm{{rot}}_{{180}}(q))={sim_q_rot:.4f}$")
        footer = rf"$\Delta=\cos(q,x_1)-\cos(q,\mathrm{{rot}}_{{180}}(q))={delta:.4f}$"

    fig.text(0.5, 0.02, footer, ha="center", va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 1])  # leave room for footer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)  # 300 dpi for thesis
    plt.close(fig)

def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    log("demo", [f"Query: {args.image}", f"Index root: {args.index_root}", f"Out: {args.out}"])

    # 1) Load store + searcher (IMPORTANT: angles=(0,) so top1 is for original orientation only)
    store = IndexStore(root=args.index_root).load_all()
    cfg = SearchConfig(
        angles=(0.0,),
        k_per_angle=64,
        crop_foreground=not args.no_crop,
        use_reranker=False,
    )
    searcher = SliceSearcher(store, cfg=cfg)

    # 2) Search top1
    img = load_image_gray(args.image)
    hits, query_img = searcher.search_image(img, k=1)
    h = hits[0]

   # 3) Reconstruct TOP1 patch image from Allen volume using manifest row
    patch_id = int(h.patch_id)

    # manifest is indexed by patch id (see store.load_all(): set_index("id"))
    row = store.manifest.loc[patch_id]

    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=int(row["resolution_um"]))

    sl = allen.get_slice(
        (float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])),
        depth=float(row["depth_vox"]),
        rotation=float(row["rotation_deg"]),
        size=int(row["slice_size_px"]),
        include_annotation=False,
    )

    x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
    top1_img = sl.image[y0:y1, x0:x1]

    if top1_img.size == 0:
        raise RuntimeError(f"Empty crop for patch_id={patch_id}: box=({x0},{y0})-({x1},{y1})")

    # 4) Rotate query 180°
    query_rot = np.rot90(query_img, 2)

    # 5) Embed + cosine for query vs rotated query
    eq = dino.embed(query_img)
    erot = dino.embed(query_rot)
    sim_q_rot = cosine(eq, erot)

    # cosine(query, top1) from the search itself
    sim_q_top1 = float(h.score)

    print(f"top1 patch_id: {patch_id}")
    print(f"cos(query, top1)          = {sim_q_top1:.6f} (from search score)")
    print(f"cos(query, rot180(query)) = {sim_q_rot:.6f}")

    # 6) Visual
    out_fr = args.out.with_name(args.out.stem + "_fr" + args.out.suffix)
    out_math = args.out.with_name(args.out.stem + "_math" + args.out.suffix)

    save_thesis_figure(
        query_img=query_img,
        top1_img=top1_img,
        query_rot=query_rot,
        sim_q_top1=sim_q_top1,
        sim_q_rot=sim_q_rot,
        out_path=out_fr,
        variant="fr",
    )

    save_thesis_figure(
        query_img=query_img,
        top1_img=top1_img,
        query_rot=query_rot,
        sim_q_top1=sim_q_top1,
        sim_q_rot=sim_q_rot,
        out_path=out_math,
        variant="math",
    )

    print(f"saved: {out_fr}")
    print(f"saved: {out_math}")


if __name__ == "__main__":
    main()