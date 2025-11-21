# index/search.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from index.store import IndexStore
from index.model.dino import model
from index.utils import log
from index.config import SLICE_SIZE


@dataclass
class SearchConfig:
    """
    Configuration for image → patch search.
    """
    angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    k_per_angle: int = 64          # how many neighbours per rotation
    crop_foreground: bool = True   # auto-crop query around tissue
    bg_threshold: float = 0.05     # pixel threshold for "foreground"
    min_fg_ratio: float = 0.05     # if less than this, skip cropping
    verbose: bool = True           # control logging (e.g. auto-crop log)


@dataclass
class SearchResult:
    """
    One retrieved patch.
    """
    patch_id: int
    score: float
    angle: float  # query rotation angle that produced this score
    meta: Dict[str, Any]  # manifest row as dict


class SliceSearcher:
    """
    High-level searcher over the patch index.

    Typical usage:
        store = IndexStore().load_all()
        searcher = SliceSearcher(store)
        hits = searcher.search_image(img_np, k=10)
    """

    def __init__(self, store: IndexStore, cfg: SearchConfig | None = None):
        self.store = store
        self.cfg = cfg or SearchConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_image(
        self,
        img_np: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Search using a query image and ALSO return the preprocessed query
        actually used for embedding (after grayscale + optional crop).

        Returns:
            hits: List[SearchResult]
            q   : np.ndarray (H,W) in [0,1]
        """
        # 1) normalize to grayscale [0,1] and optionally crop
        q, q_pil = self._prepare_query(img_np)

        # 2) build rotated views as PIL grayscale
        angles = self.cfg.angles
        pils = [q_pil.rotate(float(a), resample=Image.BILINEAR, expand=False) for a in angles]

        # 3) embed all rotations in a single batch -> (A,D)
        #    (A = number of angles)
        G = model.embed_pil_batch(pils).astype(np.float32, copy=False)  # (A, D)

        # 4) single FAISS call for all rotations at once
        #    D_all, I_all: (A, k_per_angle)
        D_all, I_all = self.store.search(G, self.cfg.k_per_angle)

        # 5) accumulate best score per patch id over all rotations
        best: Dict[int, Tuple[float, float]] = {}  # patch_id -> (score, angle)

        for a_idx, angle in enumerate(angles):
            d_row = D_all[a_idx]
            i_row = I_all[a_idx]

            for score, pid in zip(d_row, i_row):
                pid = int(pid)
                if pid < 0:
                    continue  # FAISS uses -1 for "no hit"
                if pid not in best or score > best[pid][0]:
                    best[pid] = (float(score), float(angle))

        if not best:
            return [], q

        # 6) sort by score descending and keep top-k
        sorted_hits = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        top = sorted_hits[:k]

        ids = [pid for pid, _ in top]
        df_rows = self.store.rows_for_ids(ids)

        results: List[SearchResult] = []
        for pid, (score, angle) in top:
            row = df_rows.loc[pid]
            meta = row.to_dict()
            results.append(
                SearchResult(
                    patch_id=int(pid),
                    score=float(score),
                    angle=float(angle),
                    meta=meta,
                )
            )

        return results, q

    def to_dataframe(self, hits: List[SearchResult]) -> pd.DataFrame:
        """
        Convert a list of SearchResult into a pandas DataFrame.
        """
        rows: List[Dict[str, Any]] = []
        for h in hits:
            row = dict(h.meta)
            row["patch_id"] = h.patch_id
            row["score"] = h.score
            row["query_angle_deg"] = h.angle
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Query preprocessing
    # ------------------------------------------------------------------

    def _prepare_query(self, img_np: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
        """
        Full query preprocessing pipeline:

          - convert to grayscale
          - normalize to float32 [0,1]
          - optional foreground crop
          - return both numpy (H,W) and PIL (grayscale) views
        """
        x = np.asarray(img_np)

        # Handle channels if present
        if x.ndim == 3 and x.shape[2] in (1, 3):
            if x.shape[2] == 3:
                x = x.mean(axis=2)      # RGB -> gray
            else:
                x = x[..., 0]          # (H,W,1) -> (H,W)

        if x.ndim != 2:
            raise ValueError(f"Expected (H,W) or (H,W,1/3) array, got shape {x.shape}")

        x = x.astype(np.float32, copy=False)
        if x.max() > 1.0:
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0)

        # optional auto-crop on the normalized grayscale
        if self.cfg.crop_foreground:
            x = self._auto_crop_foreground(x)

        # build PIL grayscale from the *final* query array
        x8 = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(x8, mode="L")

        return x, pil

    def _auto_crop_foreground(self, img: np.ndarray) -> np.ndarray:
        """
        Auto-crop around foreground region using a simple intensity threshold.
        If not enough foreground is found, returns the image unchanged.
        """
        cfg = self.cfg
        mask = img > float(cfg.bg_threshold)
        fg_ratio = float(mask.sum()) / float(mask.size)

        # Not enough foreground → don't crop at all.
        if fg_ratio < cfg.min_fg_ratio:
            return img

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        cropped = img[y0:y1, x0:x1]

        if cfg.verbose:
            H, W = img.shape
            ch, cw = cropped.shape
            log("search", [f"Auto-crop foreground: ({H}x{W}) -> ({ch}x{cw})"])

        return cropped
