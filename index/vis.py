# index/vis.py
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from volume.volume_helper import AllenVolume
from index.search import SearchResult
from index.config import SLICE_SIZE
from index.utils import log


def save_search_results_visuals(
    hits: List[SearchResult],
    query_img: np.ndarray,
    out_dir: Path | str,
    allen: AllenVolume | None = None,
    verbose: bool = True,
) -> None:
    """
    Save side-by-side visualizations for the top-k search hits.

    For each hit, create a 3-panel figure:
      [ query | retrieved patch | full slice with patch box ]

    Parameters
    ----------
    hits : List[SearchResult]
        Results from SliceSearcher.search_image*.
    query_img : np.ndarray
        The preprocessed query image actually used for embedding (H,W) in [0,1].
    out_dir : Path | str
        Directory where PNGs will be written.
    allen : AllenVolume | None
        If provided, reuse this volume; otherwise a new AllenVolume is created.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if allen is None:
        allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)

    if verbose:
        log("vis", [f"Saving {len(hits)} visual(s) to: {out_dir}"])


    for rank, hit in enumerate(hits, start=1):
        m = hit.meta

        normal = (float(m["normal_x"]), float(m["normal_y"]), float(m["normal_z"]))
        depth = float(m["depth_vox"])
        rot = float(m["rotation_deg"])

        x0 = int(m["x0"])
        y0 = int(m["y0"])
        x1 = int(m["x1"])
        y1 = int(m["y1"])

        # Reconstruct slice from Allen volume
        sl = allen.get_slice(
            normal=normal,
            depth=depth,
            rotation=rot,
            size=SLICE_SIZE,
            pixel=1.0,
            linear_interp=True,
            include_annotation=False,
        )
        sl_n = sl.normalized()
        full_img = sl_n.image  # (H,W) in [0,1]

        # Crop patch
        patch = full_img[y0:y1, x0:x1]

        # Build figure: [query | patch | full slice w/ rectangle]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # 1) query
        axes[0].imshow(query_img, cmap="gray")
        axes[0].set_title("Query")
        axes[0].axis("off")

        # 2) retrieved patch
        axes[1].imshow(patch, cmap="gray")
        axes[1].set_title(f"Hit {rank}\nscore={hit.score:.3f}")
        axes[1].axis("off")

        # 3) full slice with box
        axes[2].imshow(full_img, cmap="gray")
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        axes[2].add_patch(rect)
        axes[2].set_title(
            f"normal={m.get('normal_idx')} depth={m.get('depth_idx')} scale={m.get('scale')}"
        )
        axes[2].axis("off")

        fname = f"{rank:02d}_pid{hit.patch_id}_score{hit.score:.3f}.png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_gray_png(arr: np.ndarray, path: Path) -> None:
    """Save (H,W) float array in [0,1] as an 8-bit grayscale PNG."""
    arr = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    im.save(path)

def save_hits_only_images(
    hits: List[SearchResult],
    out_dir: Path | str,
    allen: AllenVolume | None = None,
    mode: str = "patch",           # "patch" or "full"
    top_n: int | None = None,
    verbose: bool = True,
) -> None:
    """
    Save plain images (no formatting) for hits, ordered by rank.

    mode="patch" saves the cropped patch (x0:y1, x0:x1).
    mode="full"  saves the reconstructed full slice.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if allen is None:
        allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)

    if top_n is None:
        top_n = len(hits)

    if verbose:
        log("hits_only", [f"Saving {min(top_n, len(hits))} image(s) to: {out_dir} (mode={mode})"])

    for rank, hit in enumerate(hits[:top_n], start=1):
        m = hit.meta

        normal = (float(m["normal_x"]), float(m["normal_y"]), float(m["normal_z"]))
        depth = float(m["depth_vox"])
        rot = float(m["rotation_deg"])

        x0 = int(m["x0"]); y0 = int(m["y0"]); x1 = int(m["x1"]); y1 = int(m["y1"])

        # Reconstruct slice from Allen volume
        sl = allen.get_slice(
            normal=normal,
            depth=depth,
            rotation=rot,
            size=SLICE_SIZE,
            pixel=1.0,
            linear_interp=True,
            include_annotation=False,
        )
        full_img = sl.normalized().image  # (H,W) in [0,1]

        if mode == "patch":
            img_out = full_img[y0:y1, x0:x1]
        elif mode == "full":
            img_out = full_img
        else:
            raise ValueError(f"Unknown mode={mode}. Use 'patch' or 'full'.")

        fname = f"hit_{rank:02d}_pid{hit.patch_id}_score{hit.score:.4f}.png"
        _save_gray_png(img_out, out_dir / fname)