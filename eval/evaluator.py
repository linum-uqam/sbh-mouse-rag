# eval/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set
import csv
import gc
import hashlib
import itertools as it
import json
import random
import time
from collections import OrderedDict, defaultdict

import numpy as np
from tqdm.auto import tqdm

from dataset.loader import MouseBrainDatasetLoader
from dataset.schema import DatasetRow
from volume.volume_helper import Slice, AllenVolume

from index.store import IndexStore
from index.search import SliceSearcher, SearchConfig, SearchResult
from index.vis import save_search_results_visuals

from eval.config import EvalConfig
from eval.stats import Stats


class _CSVStreamWriter:
    """
    Streaming CSV writer with:
      - append mode
      - lazy header initialization
      - optional header migration if new columns appear
      - periodic flush
    """
    def __init__(self, path: Path, append: bool, flush_every: int = 128) -> None:
        self.path = path
        self.append = bool(append)
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None
        self._rows_since_flush = 0

    def _read_existing_header(self) -> List[str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def _rewrite_with_merged_header(self, merged_header: List[str]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        old_rows: List[Dict[str, Any]] = []

        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    old_rows.append(row)

        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_header)
            writer.writeheader()
            for row in old_rows:
                out = {k: row.get(k, None) for k in merged_header}
                writer.writerow(out)

        tmp.replace(self.path)

    def _open_if_needed(self, rec: Dict[str, Any]) -> None:
        if self._file is not None:
            return

        rec_fields = list(rec.keys())
        existing_header = self._read_existing_header() if self.append else []

        if existing_header:
            merged = list(existing_header)
            for k in rec_fields:
                if k not in merged:
                    merged.append(k)
            if merged != existing_header:
                self._rewrite_with_merged_header(merged)
            self._fieldnames = merged
        else:
            self._fieldnames = rec_fields

        mode = "a" if (self.append and self.path.exists() and self.path.stat().st_size > 0) else "w"
        self._file = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)

        if mode == "w":
            self._writer.writeheader()
            self._file.flush()

    def write(self, rec: Dict[str, Any]) -> None:
        self._open_if_needed(rec)
        assert self._writer is not None
        assert self._fieldnames is not None

        out = {k: rec.get(k, None) for k in self._fieldnames}
        self._writer.writerow(out)
        self._rows_since_flush += 1

        if self._rows_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._fieldnames = None


@dataclass(frozen=True)
class _HitRecord:
    """Geometry-only per-hit record."""
    geom_dist_vox: float
    corner_chamfer_3pt_um: float


class Evaluator:
    """
    Dataset-driven eval loop:
      - run retrieval (optionally reranker)
      - for each hit: compute geometry distance using Slice.distance(..., physical=False)
      - build listwise soft targets over the top-K hits using adaptive tau(query)
      - write one CSV row per hit

    Notes:
      - No symmetry-aware min-over-mirror (strict laterality).
      - No region metrics, no aligned quality, no custom Chamfer implementation.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

        # --- dependencies ---
        self.store = IndexStore().load_all()

        self.search_cfg = SearchConfig(
            angles=self.cfg.angles,
            k_per_angle=self.cfg.k_per_angle,
            crop_foreground=self.cfg.crop_foreground,
            verbose=self.cfg.debug,
            use_reranker=self.cfg.use_reranker,
            rerank_topk=self.cfg.rerank_topk,
            reranker_model_path=self.cfg.reranker_model_path,
            reranker_device=self.cfg.reranker_device,
            reranker_batch_size=self.cfg.reranker_batch_size,
        )
        self.searcher = SliceSearcher(self.store, cfg=self.search_cfg)

        self.dl = MouseBrainDatasetLoader(
            csv_path=self.cfg.csv_path,
            allen_cache_dir=self.cfg.allen_cache_dir,
            allen_resolution_um=self.cfg.allen_res_um,
            size_px=self.cfg.size_px,
            pixel_step_vox=self.cfg.pixel_step_vox,
            linear_interp=self.cfg.linear_interp,
            include_annotation=False,
            real_volume_path=self.cfg.real_volume_path,
        )

        self._allen_geom = AllenVolume(
            cache_dir=self.cfg.allen_cache_dir,
            resolution_um=self.cfg.allen_res_um,
        )

        # bounded LRU cache: key=(normal_idx, depth_idx, rot_idx) -> Slice
        self._retrieved_slice_cache: OrderedDict[Tuple[int, int, int], Slice] = OrderedDict()

        # Output
        self.save_dir: Path = Path(self.cfg.save_dir or "out/eval")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv_path: Path = self.save_dir / "eval_hits.csv"

        self.eval_sig = self._make_eval_signature()

        # Safety: enforce voxel-space distance to avoid spacing mismatches
        if bool(self.cfg.distance_physical):
            raise ValueError(
                "EvalConfig.distance_physical must be False. "
                "Use voxel-space distances to avoid Allen/Real spacing mismatches."
            )

        # Resume / overwrite planning
        if self.cfg.overwrite and self.results_csv_path.exists():
            self.results_csv_path.unlink()

        self._prepare_existing_results_file()

        self._csv = _CSVStreamWriter(
            self.results_csv_path,
            append=(not self.cfg.overwrite),
            flush_every=self.cfg.csv_flush_every,
        )

        self._rows_to_process = self._compute_rows_to_process()
        self._rows_to_save = self._choose_rows_to_save(self._rows_to_process)
        self._pending_row_indices = self._build_pending_row_indices()

        # Stats
        self.stats = Stats()
        self.stats.rows_total = len(self._pending_row_indices)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self) -> None:
        if not self._pending_row_indices:
            existing_complete = self._count_completed_queries_for_current_sig()
            print(
                f"Evaluation already complete for current configuration. "
                f"Completed queries in file: {existing_complete}"
            )
            return

        pending_set = set(self._pending_row_indices)
        max_idx = max(pending_set)

        pbar = tqdm(total=len(pending_set), desc="Eval", unit="row")
        try:
            for idx, sample in enumerate(self.dl):
                if idx > max_idx:
                    break
                if idx not in pending_set:
                    continue
                self._process_dataset_row(idx=idx, sample=sample, pbar=pbar)

            pbar.close()
            self._print_summary()
        finally:
            self._csv.close()

    # -----------------------------------------------------------------
    # Resume helpers
    # -----------------------------------------------------------------

    def _make_eval_signature(self) -> str:
        payload = {
            "csv_path": str(self.cfg.csv_path),
            "source": self.cfg.source,
            "limit": self.cfg.limit,
            "size_px": self.cfg.size_px,
            "pixel_step_vox": self.cfg.pixel_step_vox,
            "linear_interp": self.cfg.linear_interp,
            "angles": list(self.cfg.angles),
            "final_k": self.cfg.final_k,
            "k_per_angle": self.cfg.k_per_angle,
            "crop_foreground": self.cfg.crop_foreground,
            "distance_grid": self.cfg.distance_grid,
            "distance_trim_frac": self.cfg.distance_trim_frac,
            "tau_q_lo": self.cfg.tau_q_lo,
            "tau_q_hi": self.cfg.tau_q_hi,
            "tau_div": self.cfg.tau_div,
            "tau_min": self.cfg.tau_min,
            "tau_max": self.cfg.tau_max,
            "use_reranker": self.cfg.use_reranker,
            "rerank_topk": self.cfg.rerank_topk,
            "reranker_model_path": str(self.cfg.reranker_model_path),
            "reranker_device": self.cfg.reranker_device,
            "reranker_batch_size": self.cfg.reranker_batch_size,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    def _prepare_existing_results_file(self) -> None:
        """
        Remove partial groups for the current signature so resume won't duplicate
        half-written queries.
        """
        if not self.results_csv_path.exists() or self.results_csv_path.stat().st_size == 0:
            return

        with self.results_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []

        # Group counts for current signature only
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for row in rows:
            sig = row.get("eval_sig", "")
            if sig != self.eval_sig:
                continue
            row_idx = row.get("row_idx")
            src = row.get("source")
            if row_idx is None or src is None:
                continue
            counts[(str(row_idx), str(src))] += 1

        partial_groups = {
            key for key, n in counts.items()
            if 0 < n < int(self.cfg.final_k)
        }

        if not partial_groups:
            return

        tmp = self.results_csv_path.with_suffix(self.results_csv_path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                sig = row.get("eval_sig", "")
                row_idx = row.get("row_idx")
                src = row.get("source")
                key = (str(row_idx), str(src))
                if sig == self.eval_sig and key in partial_groups:
                    continue
                writer.writerow(row)

        tmp.replace(self.results_csv_path)

        print(
            f"Removed {len(partial_groups)} partial query groups for current eval_sig={self.eval_sig} "
            f"before resuming."
        )

    def _scan_existing_completion_for_current_sig(self) -> Dict[int, Set[str]]:
        """
        Return:
            completed[row_idx] = {"allen", "real", ...}
        for the current eval signature only.
        """
        completed: Dict[int, Set[str]] = defaultdict(set)

        if not self.results_csv_path.exists() or self.results_csv_path.stat().st_size == 0:
            return completed

        counts: Dict[Tuple[int, str], int] = defaultdict(int)

        with self.results_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig = row.get("eval_sig", "")
                if sig != self.eval_sig:
                    continue
                try:
                    row_idx = int(row["row_idx"])
                    src = str(row["source"])
                except Exception:
                    continue
                counts[(row_idx, src)] += 1

        for (row_idx, src), n in counts.items():
            if n >= int(self.cfg.final_k):
                completed[row_idx].add(src)

        return completed

    def _expected_sources_per_row(self) -> Set[str]:
        if self.cfg.source == "allen":
            return {"allen"}
        if self.cfg.source == "real":
            return {"real"}
        return {"allen", "real"}

    def _count_completed_queries_for_current_sig(self) -> int:
        completed = self._scan_existing_completion_for_current_sig()
        expected = self._expected_sources_per_row()
        n = 0
        for _, done_sources in completed.items():
            if expected.issubset(done_sources):
                n += 1
        return n

    def _build_pending_row_indices(self) -> List[int]:
        total = self._rows_to_process
        completed = self._scan_existing_completion_for_current_sig()
        expected = self._expected_sources_per_row()

        pending: List[int] = []
        for idx in range(total):
            done_sources = completed.get(idx, set())
            if expected.issubset(done_sources):
                continue
            pending.append(idx)

        print(
            f"Eval signature: {self.eval_sig} | "
            f"total rows requested: {total} | "
            f"already complete: {total - len(pending)} | "
            f"remaining: {len(pending)}"
        )
        return pending

    # -----------------------------------------------------------------
    # Row processing
    # -----------------------------------------------------------------

    def _process_dataset_row(self, *, idx: int, sample: Dict[str, Any], pbar: tqdm) -> None:
        t0 = time.perf_counter()

        row: DatasetRow = sample["row"]
        sources = self._source_list(sample)

        row_top1_scores: List[float] = []
        row_query_latencies: List[float] = []
        row_top1_geom: List[float] = []
        row_top1_corner: List[float] = []

        for src_name, q_slice in sources:
            # skip source already completed for current signature
            if self._is_query_source_already_complete(idx=idx, src_name=src_name):
                continue

            out = self._evaluate_one_source(idx=idx, src_name=src_name, q_slice=q_slice, row=row)
            if out is None:
                continue

            hits, qimg, query_latency, hit_records, gt_tau, gt_logits, gt_probs = out
            row_query_latencies.append(query_latency)

            if not hits:
                continue

            self._write_hit_records(
                idx=idx,
                src_name=src_name,
                q_slice=q_slice,
                row=row,
                hits=hits,
                hit_records=hit_records,
                gt_tau=gt_tau,
                gt_logits=gt_logits,
                gt_probs=gt_probs,
            )

            if self.cfg.save_k and idx in self._rows_to_save:
                qdir = self.save_dir / f"{idx:05d}_{src_name}"
                qdir.mkdir(parents=True, exist_ok=True)
                save_search_results_visuals(hits, qimg, qdir, verbose=self.cfg.debug)

            top1 = hits[0]
            row_top1_corner.append(float(hit_records[0].corner_chamfer_3pt_um))
            top1_primary = (
                float(getattr(top1, "rerank_score"))
                if (self.cfg.use_reranker and getattr(top1, "rerank_score", None) is not None)
                else float(top1.score)
            )
            row_top1_scores.append(top1_primary)
            row_top1_geom.append(float(hit_records[0].geom_dist_vox))

            if self.cfg.debug:
                print(
                    f"[{idx:05d}/{src_name}] "
                    f"top1_score={float(top1.score):.4f} "
                    f"d@1_vox={float(hit_records[0].geom_dist_vox):.3f} "
                    f"tau={gt_tau:.4g}"
                )

        row_latency = time.perf_counter() - t0
        self.stats.update_row(
            row_latency_s=row_latency,
            query_latencies_s=row_query_latencies,
            row_top1_scores=row_top1_scores,
            row_top1_geom_dists=row_top1_geom,
            row_top1_corner_um=row_top1_corner,
        )

        d1 = float(np.nanmean(np.asarray(row_top1_geom, dtype=np.float64))) if row_top1_geom else float("nan")
        c1 = float(np.nanmean(np.asarray(row_top1_corner, dtype=np.float64))) if row_top1_corner else float("nan")
        pbar.set_postfix({
            "top1": f"{self.stats.avg_top1:7.4f}",
            "lat":  f"{self.stats.avg_query_latency_ms:6.0f}ms",
            "d@1":  f"{d1:7.2f}",
            "c3@1": f"{c1:7.1f}um",
        })
        pbar.update(1)

        self._maybe_collect_memory(idx)

    def _is_query_source_already_complete(self, *, idx: int, src_name: str) -> bool:
        if not self.results_csv_path.exists() or self.results_csv_path.stat().st_size == 0:
            return False

        count = 0
        with self.results_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_sig", "") != self.eval_sig:
                    continue
                try:
                    if int(row["row_idx"]) == idx and str(row["source"]) == src_name:
                        count += 1
                        if count >= int(self.cfg.final_k):
                            return True
                except Exception:
                    continue
        return False

    def _maybe_collect_memory(self, idx: int) -> None:
        if self.cfg.gc_every_rows <= 0:
            return
        if (idx + 1) % int(self.cfg.gc_every_rows) != 0:
            return

        gc.collect()

        if self.cfg.use_reranker and str(self.cfg.reranker_device).startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Source/query evaluation
    # -----------------------------------------------------------------

    def _evaluate_one_source(
        self,
        *,
        idx: int,
        src_name: str,
        q_slice: Slice,
        row: DatasetRow,
    ) -> Optional[Tuple[List[SearchResult], np.ndarray, float, List[_HitRecord], float, np.ndarray, np.ndarray]]:
        img = self._prep_img_from_slice(q_slice)

        tq0 = time.perf_counter()
        hits, qimg = self._query(img_np=img)
        query_latency = time.perf_counter() - tq0

        if not hits:
            return None

        hit_records: List[_HitRecord] = []
        for h in hits:
            d_vox = self._geom_distance_for_hit(q_slice=q_slice, hit=h)
            r_plane = self._get_retrieved_plane_slice(h)
            if r_plane is None:
                hit_records.append(_HitRecord(geom_dist_vox=float(d_vox), corner_chamfer_3pt_um=float("nan")))
                continue

            r_patch = self._crop_plane_to_patch(r_plane, h)
            if r_patch is None:
                hit_records.append(_HitRecord(geom_dist_vox=float(d_vox), corner_chamfer_3pt_um=float("nan")))
                continue

            c_um = self._corner_chamfer_3pt_um(q_slice, r_patch)
            hit_records.append(_HitRecord(
                geom_dist_vox=float(d_vox),
                corner_chamfer_3pt_um=float(c_um),
            ))

        gt_tau, gt_logits, gt_probs = self._build_soft_targets(hit_records)
        return hits, qimg, query_latency, hit_records, gt_tau, gt_logits, gt_probs

    # -----------------------------------------------------------------
    # Geometry distance
    # -----------------------------------------------------------------

    def _um_per_vox(self) -> float:
        v = self.cfg.corner_um_per_vox
        return float(v) if (v is not None) else float(self.cfg.allen_res_um)

    @staticmethod
    def _local_to_plane_px(sl: Slice, x_local: float, y_local: float) -> Tuple[float, float]:
        ox, oy = getattr(sl, "origin_px_in_plane", (0, 0))
        return float(x_local + ox), float(y_local + oy)

    def _three_points_vox_xyz(self, sl: Slice) -> np.ndarray:
        H, W = sl.image.shape
        pts_local = [(0.0, 0.0), (float(W - 1), 0.0), (0.0, float(H - 1))]

        pts = []
        for x, y in pts_local:
            xp, yp = self._local_to_plane_px(sl, x, y)
            P = sl.pixel_to_voxel(xp, yp)
            pts.append([float(P[0]), float(P[1]), float(P[2])])
        return np.asarray(pts, dtype=np.float64)

    def _corner_chamfer_3pt_um(self, a: Slice, b: Slice) -> float:
        A = self._three_points_vox_xyz(a)
        B = self._three_points_vox_xyz(b)

        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        d_ab = float(D.min(axis=1).mean())
        d_ba = float(D.min(axis=0).mean())
        chamfer_vox = 0.5 * (d_ab + d_ba)

        return chamfer_vox * self._um_per_vox()

    def _geom_distance_for_hit(self, *, q_slice: Slice, hit: SearchResult) -> float:
        r_plane = self._get_retrieved_plane_slice(hit)
        if r_plane is None:
            return float("nan")

        r_patch = self._crop_plane_to_patch(r_plane, hit)
        if r_patch is None:
            return float("nan")

        try:
            return float(Slice.distance(
                q_slice,
                r_patch,
                grid=int(self.cfg.distance_grid),
                trim=float(self.cfg.distance_trim_frac),
                physical=False,
            ))
        except Exception:
            return float("nan")

    def _get_retrieved_plane_slice(self, hit: SearchResult) -> Optional[Slice]:
        m = hit.meta or {}

        key: Optional[Tuple[int, int, int]]
        try:
            key = (int(m["normal_idx"]), int(m["depth_idx"]), int(m["rot_idx"]))
        except Exception:
            key = None

        if key is not None:
            cached = self._retrieved_slice_cache.get(key)
            if cached is not None:
                self._retrieved_slice_cache.move_to_end(key)
                return cached

        try:
            normal = (float(m["normal_x"]), float(m["normal_y"]), float(m["normal_z"]))
            depth = float(m["depth_vox"])
            rotation = float(m.get("rotation_deg", 0.0))
            size_px = int(m.get("slice_size_px", self.cfg.size_px))

            sl = self._allen_geom.get_slice(
                normal=normal,
                depth=depth,
                rotation=rotation,
                size=size_px,
                pixel=self.cfg.pixel_step_vox,
                linear_interp=self.cfg.linear_interp,
                include_annotation=False,
            )

            if key is not None and sl is not None:
                self._retrieved_slice_cache[key] = sl
                self._retrieved_slice_cache.move_to_end(key)
                while len(self._retrieved_slice_cache) > int(self.cfg.max_retrieved_slice_cache):
                    self._retrieved_slice_cache.popitem(last=False)

            return sl
        except Exception:
            return None

    @staticmethod
    def _crop_plane_to_patch(plane: Slice, hit: SearchResult) -> Optional[Slice]:
        m = hit.meta or {}
        H, W = plane.image.shape

        try:
            x0 = float(m.get("x0", 0.0))
            y0 = float(m.get("y0", 0.0))
            x1 = float(m.get("x1", float(W)))
            y1 = float(m.get("y1", float(H)))
        except Exception:
            return None

        x0 = max(0.0, min(x0, float(W)))
        x1 = max(0.0, min(x1, float(W)))
        y0 = max(0.0, min(y0, float(H)))
        y1 = max(0.0, min(y1, float(H)))

        if x1 <= x0 + 1.0 or y1 <= y0 + 1.0:
            return None

        cx_px = 0.5 * (x0 + (x1 - 1.0))
        cy_px = 0.5 * (y0 + (y1 - 1.0))

        cx = cx_px / float(max(W - 1, 1))
        cy = cy_px / float(max(H - 1, 1))

        rw = (x1 - x0) / float(max(W, 1))
        rh = (y1 - y0) / float(max(H, 1))

        try:
            return plane.crop_norm(cx=cx, cy=cy, rw=rw, rh=rh, clamp=True)
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Listwise target distribution
    # -----------------------------------------------------------------

    def _build_soft_targets(self, hit_records: List[_HitRecord]) -> Tuple[float, np.ndarray, np.ndarray]:
        K = len(hit_records)
        if K == 0:
            return 1.0, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        d = np.asarray([hr.geom_dist_vox for hr in hit_records], dtype=np.float64)
        finite = np.isfinite(d)

        if not np.any(finite):
            tau = 1.0
            logits = np.zeros((K,), dtype=np.float64)
            probs = np.full((K,), 1.0 / float(K), dtype=np.float64)
            return tau, logits, probs

        df = d[finite]
        qlo = float(np.quantile(df, float(self.cfg.tau_q_lo)))
        qhi = float(np.quantile(df, float(self.cfg.tau_q_hi)))
        spread = max(qhi - qlo, 1e-6)

        tau = spread / float(self.cfg.tau_div)
        tau = float(np.clip(tau, float(self.cfg.tau_min), float(self.cfg.tau_max)))

        logits = np.full((K,), -np.inf, dtype=np.float64)
        logits[finite] = -d[finite] / tau

        mx = float(np.max(logits[finite]))
        ex = np.zeros((K,), dtype=np.float64)
        ex[finite] = np.exp(logits[finite] - mx)
        Z = float(ex.sum())
        probs = ex / Z if Z > 0 else np.full((K,), 1.0 / float(K), dtype=np.float64)

        return tau, logits, probs

    # -----------------------------------------------------------------
    # CSV recording
    # -----------------------------------------------------------------

    def _write_hit_records(
        self,
        *,
        idx: int,
        src_name: str,
        q_slice: Slice,
        row: DatasetRow,
        hits: List[SearchResult],
        hit_records: List[_HitRecord],
        gt_tau: float,
        gt_logits: np.ndarray,
        gt_probs: np.ndarray,
    ) -> None:
        qnx, qny, qnz = q_slice.normal_xyz_unit
        q_depth = float(q_slice.depth_vox)
        q_rot = float(q_slice.rotation_deg)

        is_crop = bool(getattr(row, "is_crop", False))
        crop_area_frac = self._crop_area_frac(row)

        order_by = "rerank_score" if bool(self.cfg.use_reranker) else "score"

        for rank, (h, hr) in enumerate(zip(hits, hit_records), start=1):
            m = h.meta or {}
            rerank_score_val = getattr(h, "rerank_score", None)
            primary_score = (
                float(rerank_score_val)
                if (self.cfg.use_reranker and rerank_score_val is not None)
                else float(h.score)
            )

            rec = {
                "eval_sig": self.eval_sig,

                "row_idx": idx,
                "source": src_name,
                "rank": rank,
                "patch_id": h.patch_id,
                "patch_path": self._get_patch_path_from_hit(h),

                "score": float(h.score),
                "rerank_score": float(rerank_score_val) if rerank_score_val is not None else None,
                "order_by": order_by,
                "primary_score": primary_score,
                "query_angle_deg": float(h.angle),

                "q_normal_x": float(qnx),
                "q_normal_y": float(qny),
                "q_normal_z": float(qnz),
                "q_depth_vox": float(q_depth),
                "q_rot_deg": float(q_rot),
                "q_is_crop": bool(is_crop),
                "q_crop_area_frac": float(crop_area_frac),

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

                "corner_chamfer_3pt_um": float(hr.corner_chamfer_3pt_um),
                "geom_dist_vox": float(hr.geom_dist_vox),

                "gt_tau_vox": float(gt_tau),
                "gt_logit": float(gt_logits[rank - 1]) if (rank - 1) < int(gt_logits.size) else float("nan"),
                "gt_prob": float(gt_probs[rank - 1]) if (rank - 1) < int(gt_probs.size) else float("nan"),
            }

            self._csv.write(rec)

    # -----------------------------------------------------------------
    # Planning / sources
    # -----------------------------------------------------------------

    def _compute_rows_to_process(self) -> int:
        total = len(self.dl)
        limit = self.cfg.limit
        return total if limit is None else min(total, limit)

    def _choose_rows_to_save(self, rows_to_process: int) -> set[int]:
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

    # -----------------------------------------------------------------
    # Query/search helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _prep_img_from_slice(sl: Slice) -> np.ndarray:
        img = sl.image
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img

    def _query(self, *, img_np: np.ndarray) -> Tuple[List[SearchResult], np.ndarray]:
        hits, qimg = self.searcher.search_image(img_np=img_np, k=self.cfg.final_k)
        return hits or [], qimg

    # -----------------------------------------------------------------
    # Summary / misc
    # -----------------------------------------------------------------

    def _print_summary(self) -> None:
        s = self.stats
        print("\n=== Summary (per row) ===")
        print(f"Rows processed     : {s.rows_done}")
        if s.rows_done:
            print(f"Avg Top-1 score    : {s.avg_top1:.4f}")
            print(f"Avg query latency  : {s.avg_query_latency_ms:.0f} ms (per search_image call)")
            print(f"Avg row latency    : {s.avg_row_latency_ms:.0f} ms (full pipeline)")
        if s.rows_with_geom:
            print(f"Avg geom dist @1   : {s.avg_geom_dist:.2f} vox (Slice.distance, physical=False)")
        if s.rows_with_corner:
            print(f"Avg corner@1       : {s.avg_corner_um:.1f} um (3pt corner chamfer)")

    @staticmethod
    def _get_patch_path_from_hit(h: SearchResult) -> str:
        p = getattr(h, "path", None)
        if p:
            return str(p)

        m = getattr(h, "meta", {}) or {}
        for k in ("patch_path", "img_path", "path", "rel_path", "png_path"):
            v = m.get(k)
            if v:
                return str(v)
        return ""

    @staticmethod
    def _crop_area_frac(row: DatasetRow) -> float:
        is_crop = bool(getattr(row, "is_crop", False))
        if not is_crop:
            return 1.0
        try:
            rw = float(getattr(row, "crop_rw"))
            rh = float(getattr(row, "crop_rh"))
            return float(max(0.0, min(1.0, rw * rh)))
        except Exception:
            return float("nan")