# eval/evaluator.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import time
import numpy as np
from tqdm.auto import tqdm
import random
import pandas as pd
import itertools as it

from dataset.loader import MouseBrainDatasetLoader
from dataset.schema import DatasetRow
from volume.volume_helper import Slice, AllenVolume

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig, SearchResult
from index.vis import save_search_results_visuals
from eval.metrics import (
    compute_spatial_error,
    compute_region_error,
    SpatialError,
)
from eval.config import EvalConfig
from eval.stats import Stats

class Evaluator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

        # --- build deps once ---
        self.store = IndexStore().load_all()

        self.search_cfg = SearchConfig(
            angles=self.cfg.angles,
            k_per_angle=self.cfg.k_per_angle,
            crop_foreground=self.cfg.crop_foreground,
            verbose=self.cfg.debug,
        )
        self.searcher = SliceSearcher(self.store, cfg=self.search_cfg)

        self.dl = MouseBrainDatasetLoader(
            csv_path=self.cfg.csv_path,
            allen_cache_dir=self.cfg.allen_cache_dir,
            allen_resolution_um=self.cfg.allen_res_um,
            size_px=self.cfg.size_px,
            pixel_step_vox=self.cfg.pixel_step_vox,
            linear_interp=self.cfg.linear_interp,
            include_annotation=self.cfg.include_annotation,
            real_volume_path=self.cfg.real_volume_path,
        )

        # Allen volume for labels
        self._allen_labels = AllenVolume(
            cache_dir=self.cfg.allen_cache_dir,
            resolution_um=self.cfg.allen_res_um,
        )
        self._allen_labels.normalize_volume()

        self.stats = Stats()
        self.stats.rows_total = len(self.dl)

        # --- output paths ---
        # Always normalize save_dir to a Path, even if user passed None.
        self.save_dir: Path = Path(self.cfg.save_dir or "out/eval")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # where eval_hits.csv goes
        self.results_csv_path: Path = self.save_dir / "eval_hits.csv"

        # collected per-hit records
        self._records: List[Dict[str, Any]] = []

        # planning: how many rows we process, which ones to visualize
        self._rows_to_process = self._compute_rows_to_process()
        self._rows_to_save = self._choose_rows_to_save(self._rows_to_process)

        # voxel size in microns in the slice plane (approximate)
        self._voxel_size_um = float(self.cfg.allen_res_um) * float(self.cfg.pixel_step_vox)

    # ---------------- convenience properties over cfg ----------------
    @property
    def source(self) -> str:
        return self.cfg.source

    @property
    def size_px(self) -> int:
        return self.cfg.size_px

    @property
    def final_k(self) -> int:
        return self.cfg.final_k

    @property
    def pixel_step_vox(self) -> float:
        return self.cfg.pixel_step_vox

    @property
    def include_annotation(self) -> bool:
        return self.cfg.include_annotation

    @property
    def debug(self) -> bool:
        return self.cfg.debug

        # ---------------- public API ----------------
    def run(self) -> None:
        total = self._rows_to_process
        pbar = tqdm(total=total, desc="Eval", unit="row")

        for idx, sample in enumerate(it.islice(self.dl, total)):
            t0 = time.perf_counter()

            row: DatasetRow = sample["row"]
            src_list = self._source_list(sample)  # [('allen', Slice), ('real', Slice?)]

            row_top1_scores: List[float] = []
            row_spatial_errors: List[SpatialError] = []
            row_region_errors: List[float] = []
            row_query_latencies: List[float] = []  # <-- per-query search times

            for src_name, sl in src_list:
                img = self._prep_img_from_slice(sl)

                # --- pure query latency: only SliceSearcher.search_image ---
                tq0 = time.perf_counter()
                hits, qimg = self._query(img)
                query_latency = time.perf_counter() - tq0
                row_query_latencies.append(query_latency)

                if not hits:
                    continue

                ranked_hits: List[SearchResult] = hits

                # Per-hit metrics for this (row, source)
                per_hit_spatial: List[SpatialError] = []
                per_hit_region: List[float] = []

                for h in ranked_hits:
                    se = compute_spatial_error(
                        q_slice=sl,
                        hit=h,
                    )
                    per_hit_spatial.append(se)

                    labels_q = sl.labels if self.include_annotation else None
                    labels_r = self._sample_retrieved_patch_labels(h)
                    re = compute_region_error(labels_q, labels_r)
                    per_hit_region.append(re)

                # top-1 metrics for stats (per source)
                top1 = ranked_hits[0]
                row_top1_scores.append(float(top1.score))
                row_spatial_errors.append(per_hit_spatial[0])
                row_region_errors.append(per_hit_region[0])

                # Logging (top-1 only)
                self._log_hits(
                    idx,
                    src_name,
                    sl,
                    ranked_hits,
                    row,
                    per_hit_spatial[0],
                    per_hit_region[0],
                )

                # Record all ranks for CSV
                self._record_hits(
                    idx,
                    src_name,
                    sl,
                    ranked_hits,
                    row,
                    per_hit_spatial,
                    per_hit_region,
                )

                # Optional image saving
                if self.cfg.save_k and idx in self._rows_to_save:
                    qdir = self.save_dir / f"{idx:05d}_{src_name}"
                    qdir.mkdir(parents=True, exist_ok=True)
                    save_search_results_visuals(
                        ranked_hits,
                        qimg,
                        qdir,
                        verbose=self.cfg.debug,
                    )

            # full row latency (for information; not what we show as "lat")
            row_latency = time.perf_counter() - t0

            self.stats.update_row(
                row_latency_s=row_latency,
                query_latencies_s=row_query_latencies,
                row_top1_scores=row_top1_scores,
                row_spatial_errors=row_spatial_errors,
                row_region_errors=row_region_errors,
            )

            pbar.set_postfix({
                "top1": f"{self.stats.avg_top1:7.4f}",
                "lat": f"{self.stats.avg_query_latency_ms:6.0f}ms",
                "sp_err": f"{self.stats.avg_spatial_dist:7.2f}",
                "reg_l1": f"{self.stats.avg_region_l1_error * 100:6.2f}%",  # % mismatch
            })
            pbar.update(1)

        pbar.close()
        self._print_summary()

        if self.results_csv_path and self._records:
            df = pd.DataFrame(self._records)
            self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.results_csv_path, index=False)
            print(f"Detailed results CSV saved to: {self.results_csv_path}")


    # ---------------- internals ----------------
    def _compute_rows_to_process(self) -> int:
        """How many dataset rows should we actually evaluate?"""
        total = len(self.dl)
        limit = self.cfg.limit
        return total if limit is None else min(total, limit)

    def _choose_rows_to_save(self, rows_to_process: int) -> set[int]:
        """
        Decide which row indices (0-based) we will save visuals for.
        We only save examples if save_k is set and > 0.
        """
        k = self.cfg.save_k
        if not k or k <= 0:
            return set()

        k = min(k, rows_to_process)
        rng = random.Random(self.cfg.save_seed)
        return set(rng.sample(range(rows_to_process), k))
    
    def _source_list(self, sample: Dict[str, Any]) -> List[Tuple[str, Slice]]:
        source = self.cfg.source
        allen_sl: Slice = sample["allen"]
        real_sl: Optional[Slice] = sample.get("real")

        out: List[Tuple[str, Slice]] = []
        if source in ("allen", "both"):
            out.append(("allen", allen_sl))
        if source in ("real", "both") and real_sl is not None:
            out.append(("real", real_sl))
        return out

    @staticmethod
    def _prep_img_from_slice(sl: Slice) -> np.ndarray:
        img = sl.image
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img

    def _query(self, img_np: np.ndarray) -> Tuple[List[SearchResult], np.ndarray]:
        hits, qimg = self.searcher.search_image(
            img_np=img_np,
            k=self.final_k,
        )
        return hits or [], qimg

    def _sample_retrieved_patch_labels(self, hit: SearchResult) -> Optional[np.ndarray]:
        """
        Sample Allen annotations for the retrieved patch.
        """
        if not self.include_annotation:
            return None

        m = hit.meta
        try:
            normal = (
                float(m["normal_x"]),
                float(m["normal_y"]),
                float(m["normal_z"]),
            )
            depth = float(m["depth_vox"])
            rotation = float(m.get("rotation_deg", 0.0))
            size_px = int(m.get("size_px", self.size_px))

            sl = self._allen_labels.get_slice(
                normal=normal,
                depth=depth,
                rotation=rotation,
                size=size_px,
                pixel=self.pixel_step_vox,
                linear_interp=True,
                include_annotation=True,
            )
            if sl.labels is None:
                return None

            x0 = int(round(float(m.get("x0", 0))))
            y0 = int(round(float(m.get("y0", 0))))
            x1 = int(round(float(m.get("x1", sl.labels.shape[1]))))
            y1 = int(round(float(m.get("y1", sl.labels.shape[0]))))

            H, W = sl.labels.shape
            x0 = max(0, min(x0, W))
            x1 = max(0, min(x1, W))
            y0 = max(0, min(y0, H))
            y1 = max(0, min(y1, H))
            if x1 <= x0 or y1 <= y0:
                return None

            return sl.labels[y0:y1, x0:x1]
        except Exception:
            return None

    def _log_hits(
        self,
        idx: int,
        src_name: str,
        sl: Slice,
        hits: List[SearchResult],
        row: DatasetRow,
        se: SpatialError,
        re: float,
    ) -> None:
        if not self.debug:
            return

        nlog = tuple(round(x, 3) for x in sl.normal_xyz_unit)
        print(
            f"[{idx:05d}/{src_name}] "
            f"n={nlog} depth={sl.depth_vox:.1f} rot={sl.rotation_deg:.1f}"
        )
        print(
            f"  spatial_dist={se.dist:.3f} vox  "
            f"d=({se.dx:.1f},{se.dy:.1f},{se.dz:.1f})  "
            f"region_l1_error={re:.3f}"
        )

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
        print(f"Rows processed     : {s.rows_done}")
        if s.rows_done:
            print(f"Avg Top-1 score    : {s.avg_top1:.4f}")
            print(f"Avg query latency  : {s.avg_query_latency_ms:.0f} ms "
                  f"(per search_image call)")
            print(f"Avg row latency    : {s.avg_row_latency_ms:.0f} ms "
                  f"(full pipeline)")
        if s.rows_with_spatial:
            avg_vox = s.avg_spatial_dist
            avg_um = avg_vox * self._voxel_size_um
            print(
                f"Avg spatial error  : {avg_vox:.2f} vox "
                f"({avg_um:.2f} µm, approx)"
            )
        if s.rows_with_region:
            print(
                f"Avg region mismatch: {s.avg_region_l1_error * 100:.2f}% "
                f"(region-pixel mismatch, L1)"
            )

    def _record_hits(
        self,
        idx: int,
        src_name: str,
        sl: Slice,
        hits: List[SearchResult],
        row: DatasetRow,
        spatial_list: List[SpatialError],
        region_list: List[float],
    ) -> None:
        """
        Append one record per (row, hit) to self._records for later CSV export.
        We also store per-row metadata so we can aggregate later.

        Metrics stored:
          - spatial_dist_vox, spatial_dist_um, dx, dy, dz
          - region_l1_error  (fraction of pixels with mismatched region)
        """
        qnx, qny, qnz = sl.normal_xyz_unit
        q_depth = float(sl.depth_vox)
        q_rot = float(sl.rotation_deg)

        is_crop = bool(row.is_crop)

        # Compute area fraction from linear fractions if this is a crop.
        # For full slices, define area_frac = 1.0.
        if is_crop:
            try:
                rw = float(row.crop_rw)
                rh = float(row.crop_rh)
                crop_area_frac = max(0.0, min(1.0, rw * rh))
            except Exception:
                crop_area_frac = float("nan")
        else:
            crop_area_frac = 1.0

        for rank, (h, se, re) in enumerate(
            zip(hits, spatial_list, region_list),
            start=1,
        ):
            m = h.meta
            spatial_dist_vox = float(se.dist)
            spatial_dist_um = spatial_dist_vox * self._voxel_size_um

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
                "q_is_crop": is_crop,
                "q_crop_area_frac": crop_area_frac,
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
                # spatial metrics (for this rank)
                "spatial_dist_vox": spatial_dist_vox,
                "spatial_dist_um": spatial_dist_um,
                "spatial_dx_vox": se.dx,
                "spatial_dy_vox": se.dy,
                "spatial_dz_vox": se.dz,
                # region metric (for this rank)
                "region_l1_error": re,
            }

            self._records.append(rec)