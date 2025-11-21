# eval/evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import time
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from volume.volume_helper import Slice
from dataset import MouseBrainDatasetLoader  # new dataset API

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig, SearchResult
from index.vis import save_search_results_visuals


class RunningStats:
    def __init__(self) -> None:
        self.rows_total = 0
        self.rows_done = 0
        self.sum_top1_score = 0.0
        self.sum_latency_s = 0.0

    def update_row(
        self,
        latency_s: float,
        row_top1_scores: List[float],
    ) -> None:
        self.rows_done += 1
        self.sum_latency_s += latency_s
        if row_top1_scores:
            self.sum_top1_score += float(np.nanmean(row_top1_scores))

    @property
    def avg_top1(self) -> float:
        return (self.sum_top1_score / self.rows_done) if self.rows_done else float("nan")

    @property
    def avg_latency_ms(self) -> float:
        return ((self.sum_latency_s / self.rows_done) * 1000.0) if self.rows_done else 0.0


class Evaluator:
    """
    Stateful evaluator:
      - Holds all knobs as attributes
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

        # Slice sampling (eval-time)
        size_px: int = 512,
        pixel_step_vox: float = 1.0,
        linear_interp: bool = False,                        # False for faster; True for better

        # Search
        angles: Tuple[float, ...] = (0, 90, 180, 270),
        final_k: int = 10,
        k_per_angle: int = 64,
        crop_foreground: bool = True,
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

        self.angles = angles
        self.final_k = final_k
        self.k_per_angle = k_per_angle
        self.crop_foreground = crop_foreground
        self.debug = debug

        self.save_dir = save_dir
        self.save_k = save_k
        self.save_seed = save_seed

        # --- build deps once: index + searcher ---
        self.store = IndexStore().load_all()

        self.search_cfg = SearchConfig(
            angles=self.angles,
            k_per_angle=self.k_per_angle,
            crop_foreground=self.crop_foreground,
            verbose=self.debug,
        )
        self.searcher = SliceSearcher(self.store, cfg=self.search_cfg)

        # --- dataset loader (new dataset API) ---
        self.dl = MouseBrainDatasetLoader(
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

        # --- output paths ---
        self.out_root = Path(self.save_dir) if self.save_dir else None
        if self.out_root:
            self.out_root.mkdir(parents=True, exist_ok=True)
            self.results_csv_path = self.out_root / "eval_hits.csv"
        else:
            self.results_csv_path = None

        # collect detailed per-hit records for CSV
        self._records: List[Dict[str, Any]] = []

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

            for src_name, sl in src_list:
                img = self._prep_img_from_slice(sl)
                hits, qimg = self._query(img)
                if not hits:
                    continue

                # top-1 score
                top1 = hits[0]
                row_top1_scores.append(float(top1.score))

                # Logging (only if debug=True)
                self._log_hits(idx, src_name, sl, hits)

                # Record hits for CSV
                self._record_hits(idx, src_name, sl, hits)

                # Optional image saving
                if self.out_root and idx in self._rows_to_save:
                    qdir = self.out_root / f"{idx:05d}_{src_name}"
                    qdir.mkdir(parents=True, exist_ok=True)
                    save_search_results_visuals(
                        hits,
                        qimg,
                        qdir,
                        verbose=self.debug,
                    )

            latency = time.perf_counter() - t0
            self.stats.update_row(
                latency_s=latency,
                row_top1_scores=row_top1_scores,
            )

            pbar.set_postfix({
                "top1": f"{self.stats.avg_top1:7.4f}",
                "lat": f"{self.stats.avg_latency_ms:6.0f}ms",
            })
            pbar.update(1)

        pbar.close()
        self._print_summary()

        # Write detailed per-hit CSV (if requested)
        if self.results_csv_path and self._records:
            df = pd.DataFrame(self._records)
            self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.results_csv_path, index=False)
            print(f"Detailed results CSV saved to: {self.results_csv_path}")

    # ---------------- internals ----------------
    def _source_list(self, sample: Dict[str, Any]) -> List[Tuple[str, Slice]]:
        """
        Map loader sample -> list of (source_name, Slice) to evaluate.
        Loader is expected to return {"allen": Slice, "real": Optional[Slice], "row": DatasetRow}.
        """
        source = self.source
        allen_sl: Slice = sample["allen"]
        real_sl: Optional[Slice] = sample.get("real")

        out: List[Tuple[str, Slice]] = []
        if source in ("allen", "both"):
            out.append(("allen", allen_sl))
        if source in ("real", "both") and real_sl is not None:
            out.append(("real", real_sl))
        return out

    def _set_random_rows(self) -> None:
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

    def _query(self, img_np: np.ndarray) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Run search and return (hits, query_img_used).
        """
        hits, qimg = self.searcher.search_image(
            img_np=img_np,
            k=self.final_k,
        )
        return hits or [], qimg

    def _log_hits(
        self,
        idx: int,
        src_name: str,
        sl: Slice,
        hits: List[SearchResult],
    ) -> None:
        # If not debugging, stay silent
        if not self.debug:
            return

        nlog = tuple(round(x, 3) for x in sl.normal_xyz_unit)
        print(f"[{idx:05d}/{src_name}] n={nlog} depth={sl.depth_vox:.1f} rot={sl.rotation_deg:.1f}")
        N = min(5, len(hits))
        for rank, h in enumerate(hits[:N], start=1):
            m = h.meta
            n = (
                m.get("normal_x", 0.0),
                m.get("normal_y", 0.0),
                m.get("normal_z", 0.0),
            )
            cols = [
                f"{rank:02d} pid={h.patch_id}",
                f"score={h.score:.4f}",
                f"q_angle={h.angle:5.1f}°",
                f"n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})",
                f"depth={m.get('depth_vox', float('nan')):.1f}",
                f"scale={m.get('scale', '?')}",
                f"box=({m.get('x0','?')},{m.get('y0','?')})-({m.get('x1','?')},{m.get('y1','?')})",
            ]
            print("  " + "  ".join(cols))

    def _print_summary(self) -> None:
        s = self.stats
        print("\n=== Summary (per row) ===")
        print(f"Rows processed : {s.rows_done}")
        if s.rows_done:
            print(f"Avg Top-1 score: {s.avg_top1:.4f}")
            print(f"Avg latency    : {s.avg_latency_ms:.0f} ms")

    def _record_hits(
        self,
        idx: int,
        src_name: str,
        sl: Slice,
        hits: List[SearchResult],
    ) -> None:
        """
        Append one record per (row, hit) to self._records for later CSV export.
        """
        qnx, qny, qnz = sl.normal_xyz_unit
        q_depth = float(sl.depth_vox)
        q_rot = float(sl.rotation_deg)

        for rank, h in enumerate(hits, start=1):
            m = h.meta
            rec = {
                "row_idx": idx,
                "source": src_name,
                "rank": rank,
                "patch_id": h.patch_id,
                "score": h.score,
                "query_angle_deg": h.angle,
                # query pose
                "q_normal_x": float(qnx),
                "q_normal_y": float(qny),
                "q_normal_z": float(qnz),
                "q_depth_vox": q_depth,
                "q_rot_deg": q_rot,
                # retrieved patch pose / metadata
                "normal_idx": m.get("normal_idx"),
                "depth_idx": m.get("depth_idx"),
                "rot_idx": m.get("rot_idx"),
                "normal_x": m.get("normal_x"),
                "normal_y": m.get("normal_y"),
                "normal_z": m.get("normal_z"),
                "depth_vox": m.get("depth_vox"),
                "rotation_deg": m.get("rotation_deg"),
                "scale": m.get("scale"),
                "patch_row": m.get("patch_row"),
                "patch_col": m.get("patch_col"),
                "x0": m.get("x0"),
                "y0": m.get("y0"),
                "x1": m.get("x1"),
                "y1": m.get("y1"),
                "patch_h": m.get("patch_h"),
                "patch_w": m.get("patch_w"),
            }
            self._records.append(rec)
