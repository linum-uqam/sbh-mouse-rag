# eval/evaluator.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import time
import numpy as np
from tqdm.auto import tqdm

from dataset.data_loader import DataLoader
from volume.volume_helper import Slice

from index.store import IndexStore
from index.searcher import SliceSearcher
from index.visual import save_results_images
import random

class RunningStats:
    def __init__(self) -> None:
        self.rows_total = 0
        self.rows_done = 0
        self.sum_top1_score = 0.0
        self.sum_latency_s = 0.0
        self.sum_top1_delta_col = 0.0
        self.rows_with_col = 0

    def update_row(
        self,
        latency_s: float,
        row_top1_scores: List[float],
        row_top1_deltas: List[float],
    ) -> None:
        self.rows_done += 1
        self.sum_latency_s += latency_s
        if row_top1_scores:
            self.sum_top1_score += float(np.nanmean(row_top1_scores))
        if row_top1_deltas:
            self.sum_top1_delta_col += float(np.nanmean(row_top1_deltas))
            self.rows_with_col += 1

    @property
    def avg_top1(self) -> float:
        return (self.sum_top1_score / self.rows_done) if self.rows_done else float("nan")

    @property
    def avg_latency_ms(self) -> float:
        return ((self.sum_latency_s / self.rows_done) * 1000.0) if self.rows_done else 0.0

    @property
    def avg_delta_col(self) -> float:
        return (self.sum_top1_delta_col / self.rows_with_col) if self.rows_with_col else 0.0


class Evaluator:
    """
    Stateful evaluator:
      - Holds all knobs as attributes (no separate config object)
      - Owns searcher, dataloader, stats, logging, and run() loop
    """

    def __init__(
        self,
        *,
        # Data / dataset
        csv_path: str = "dataset/dataset.csv",
        source: str = "allen",                              # "allen" | "real" | "both"
        limit: Optional[int] = None,
        include_annotation: bool = True,

        # Volumes
        allen_cache_dir: str = "volume/data/allen",
        allen_res_um: int = 25,
        real_volume_path: Optional[str] = "volume/data/real/real_mouse_brain_ras_25um.nii.gz",

        # Slice sampling
        size_px: int = 512,
        pixel_step_vox: float = 1.0,
        linear_interp: bool = False,                        # False for faster results, and True for better results. 

        # Index/search
        mode: str = "col",                                  # "base" | "col"
        angles: Tuple[float, ...] = (0, 90, 180, 270),
        scales: Tuple[int, ...] = (1, 2, 4, 8, 14),
        final_k: int = 10,
        base_per_rotation: int = 200,
        base_topk_for_col: int = 300,
        use_token_ann: bool = True,
        token_scales: Tuple[int, ...] = (4, 8, 14),
        token_topM: int = 32,
        debug: bool = False,

        # Output
        save_dir: Optional[str] = None,
        save_k: Optional[int] = None,
        save_seed: int = 42,
    ):
        # --- store knobs as attributes ---
        self.csv_path = csv_path
        self.source = source
        self.limit = limit
        self.include_annotation = include_annotation

        self.allen_cache_dir = allen_cache_dir
        self.allen_res_um = allen_res_um
        self.real_volume_path = real_volume_path

        self.size_px = size_px
        self.pixel_step_vox = pixel_step_vox
        self.linear_interp = linear_interp

        self.mode = mode
        self.angles = angles
        self.scales = scales
        self.final_k = final_k
        self.base_per_rotation = base_per_rotation
        self.base_topk_for_col = base_topk_for_col
        self.use_token_ann = use_token_ann
        self.token_scales = token_scales
        self.token_topM = token_topM
        self.debug = debug
        
        self.save_dir = save_dir
        self.save_k = save_k
        self.save_seed = save_seed

        # --- build deps once ---
        self.store = IndexStore().load_all()
        self.searcher = SliceSearcher(self.store)

        self.dl = DataLoader(
            csv_path=self.csv_path,
            allen_cache_dir=self.allen_cache_dir,
            allen_resolution_um=self.allen_res_um,
            size_px=self.size_px,
            pixel_step_vox=self.pixel_step_vox,
            linear_interp=self.linear_interp,
            include_annotation=self.include_annotation,
            real_volume_path=self.real_volume_path,
        )

        self.stats = RunningStats()
        self.stats.rows_total = len(self.dl)

        self.out_root = Path(self.save_dir) if self.save_dir else None
        if self.out_root:
            self.out_root.mkdir(parents=True, exist_ok=True)

        self._set_random_rows() 

    # ---------------- public API ----------------
    def run(self) -> None:
        total = (len(self.dl) if self.limit is None else min(self.limit, len(self.dl)))
        pbar = tqdm(total=total, desc="Eval", unit="row")

        for idx, sample in enumerate(self.dl):
            if self.limit is not None and self.stats.rows_done >= self.limit:
                break

            t0 = time.perf_counter()

            src_list = self._source_list(sample)   # [('allen', Slice), ('real', Slice?)]
            row_top1_scores: List[float] = []
            row_top1_deltas: List[float] = []

            for src_name, sl in src_list:
                img = self._prep_img_from_slice(sl)
                hits = self._query(img)
                if not hits:
                    continue

                for r, h in enumerate(hits, 1):
                    h["rank"] = r

                self._log_hits(idx, src_name, sl, hits)

                top1 = hits[0]
                row_top1_scores.append(float(top1.get("score", float("nan"))))
                if "col_score" in top1 and "base_score" in top1:
                    row_top1_deltas.append(float(top1["col_score"] - top1["base_score"]))

                if self.out_root and idx in self._rows_to_save:
                    qdir = self.out_root / f"{idx:05d}_{src_name}"
                    qdir.mkdir(parents=True, exist_ok=True)
                    save_results_images(hits, qdir)

            latency = time.perf_counter() - t0
            self.stats.update_row(
                latency_s=latency,
                row_top1_scores=row_top1_scores,
                row_top1_deltas=row_top1_deltas,
            )

            pbar.set_postfix({
                "top1": f"{self.stats.avg_top1:7.4f}",
                "Δcol": f"{self.stats.avg_delta_col:+7.4f}",
                "lat": f"{self.stats.avg_latency_ms:6.0f}ms",
            })
            pbar.update(1)

        pbar.close()
        self._print_summary()

    # ---------------- internals ----------------
    def _source_list(self, sample: Dict[str, Any]) -> List[Tuple[str, Slice]]:
        source = self.source
        allen_sl: Slice = sample["allen"]
        real_sl: Optional[Slice] = sample["real"]

        out: List[Tuple[str, Slice]] = []
        if source in ("allen", "both"):
            out.append(("allen", allen_sl))
        if source in ("real", "both") and real_sl is not None:
            out.append(("real", real_sl))
        return out
    
    def _set_random_rows(self):
        self._rows_to_process = (len(self.dl) if self.limit is None else min(self.limit, len(self.dl)))
        self._rows_to_save: set[int] = set()
        if self.out_root and self.save_k is not None and self.save_k > 0:
            k = min(self.save_k, self._rows_to_process)
            random.seed(self.save_seed)
            self._rows_to_save = set(random.sample(range(self._rows_to_process), k))

    @staticmethod
    def _prep_img_from_slice(sl: Slice) -> np.ndarray:
        # Volumes are normalized, so just ensure float32
        img = sl.image
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img

    def _query(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        return self.searcher.search_image(
            img_np=img_np,
            angles=list(self.angles),
            scales=list(self.scales),
            mode=self.mode,
            base_topk_for_col=self.base_topk_for_col,
            base_per_rotation=self.base_per_rotation,
            token_topM=self.token_topM,
            token_scales=list(self.token_scales),
            use_token_ann=self.use_token_ann,
            rerank_final_k=self.final_k,
            debug=self.debug,
        ) or []

    def _log_hits(self, idx: int, src_name: str, sl: Slice, hits: List[Dict[str, Any]]) -> None:
        nlog = tuple(round(x, 3) for x in sl.normal_xyz_unit)
        print(f"[{idx:05d}/{src_name}] n={nlog} depth={sl.depth_vox:.1f} rot={sl.rotation_deg:.1f}")
        N = min(5, len(hits))
        for h in hits[:N]:
            n = h["normal"]
            cols = [
                f"{h['rank']:02d} id={h['slice_id']}",
                f"score={h['score']:.4f}",
                f"base={h.get('base_score', float('nan')):.4f}",
                f"best_rot={h.get('best_angle', 0.0):>6.1f}°",
            ]
            if "col_score" in h:
                cols.append(f"col={h['col_score']:.4f}")
            if "best_scale" in h:
                cols.append(f"scale={h['best_scale']}x{h['best_scale']}")
            cols.extend([
                f"n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})",
                f"depth={h['depth_vox']:.1f}",
            ])
            print("  " + "  ".join(cols))

    def _print_summary(self) -> None:
        s = self.stats
        print("\n=== Summary (per row) ===")
        print(f"Rows processed : {s.rows_done}")
        if s.rows_done:
            print(f"Avg Top-1 score: {s.avg_top1:.4f}")
            print(f"Avg latency    : {s.avg_latency_ms:.0f} ms")
            if s.rows_with_col:
                print(f"Avg Δ(col-base) on Top-1 (rows w/ col): {s.avg_delta_col:+.4f}")
