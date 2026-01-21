# index/patch_index.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from volume.volume_helper import VolumeHelper
from index.geom import iter_slices_fibonacci, count_slices_fibonacci
from index.model.dino import model
from index.utils import log
from .config import (
    K_NORMALS,
    SLICE_SIZE,
    PATCH_SCALES,
    PATCH_OVERLAP,
    D,
    INDEX_STRATEGY,
    AUTO_FLAT_MAX,
    AUTO_HNSW_MAX,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    PQ_M,
    PQ_BITS,
)


# ----------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------

@dataclass
class PatchSamplingConfig:
    """
    Configuration for how we sample patches from each slice.
    """
    slice_size_px: int = SLICE_SIZE
    patch_scales: Tuple[int, ...] = PATCH_SCALES
    patch_overlap: float = PATCH_OVERLAP
    bg_threshold: float = 0.05       # pixel intensity threshold for foreground
    min_fg_ratio: float = 0.05       # minimum foreground ratio for a patch
    resolution_um: int = 25          # for metadata only
    batch_size: int = 64             # batch size for DINO embeddings


@dataclass
class IndexConfig:
    """
    Configuration for the FAISS index strategy.
    """
    dim: int = D
    strategy: str = INDEX_STRATEGY      # "auto" | "flat" | "hnsw" | "ivfpq"
    auto_flat_max: int = AUTO_FLAT_MAX
    auto_hnsw_max: int = AUTO_HNSW_MAX
    hnsw_m: int = HNSW_M
    hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION
    pq_m: int = PQ_M
    pq_bits: int = PQ_BITS
    max_train_vectors: int = 200_000   # cap for IVFPQ training sample

@dataclass
class PatchMeta:
    """
    All metadata needed to later build the manifest row for a single patch.
    """
    # slice pose (3D)
    normal_idx: int
    depth_idx: int
    rot_idx: int
    normal_x: float
    normal_y: float
    normal_z: float
    depth_vox: float
    rotation_deg: float

    # patch-level info
    scale: int
    patch_row: int
    patch_col: int
    x0: int
    y0: int
    x1: int
    y1: int
    patch_h: int
    patch_w: int

    # global slice info
    slice_size_px: int
    resolution_um: int

    # patch center in voxel space (one logical field)
    center_xyz_vox: tuple[float, float, float]

# ----------------------------------------------------------------------
# FAISS index manager
# ----------------------------------------------------------------------

class FaissIndexManager:
    """
    Responsible for:
      - choosing the index strategy (flat / hnsw / ivfpq)
      - building and training the FAISS index
      - adding vectors with ids

    Public API:
      - build_index(vecs, ids) -> faiss.Index
    """

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg

    # ---------- strategy choice ----------

    def _choose_strategy(self, n_vecs: int) -> str:
        """Return 'flat' | 'hnsw' | 'ivfpq'."""
        s = (self.cfg.strategy or "auto").lower()
        if s != "auto":
            return s

        if n_vecs <= self.cfg.auto_flat_max:
            return "flat"
        if n_vecs <= self.cfg.auto_hnsw_max:
            return "hnsw"
        return "ivfpq"

    # ---------- helpers ----------

    @staticmethod
    def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
        """
        L2-normalize rows in-place (cosine -> inner product).
        """
        X = vecs.astype(np.float32, copy=False)
        faiss.normalize_L2(X)
        return X

    @staticmethod
    def _choose_nlist(n_train: int) -> int:
        """
        nlist ≈ n_train / 40, clamped to [256, 65536] and multiple of 128.
        """
        raw = max(1, n_train // 40)
        n = max(256, raw)
        nlist = (n // 128) * 128
        return min(nlist, 65536)

    # ---------- builder entrypoint ----------

    def build_index(self, vecs: np.ndarray, ids: np.ndarray) -> faiss.Index:
        """
        vecs: (N,D) float32
        ids : (N,) int64
        """
        N, d = vecs.shape
        if d != self.cfg.dim:
            raise ValueError(f"Embedding dimension mismatch: vecs dim={d}, cfg.dim={self.cfg.dim}")

        X = self._normalize_rows(vecs)
        I = np.asarray(ids, dtype=np.int64)
        strategy = self._choose_strategy(N)

        log("index", [
            f"Total vectors: {N}",
            f"Dim (D)      : {d}",
            f"Strategy     : {strategy} (requested={self.cfg.strategy})",
        ])

        builders = {
            "flat": self._build_flat,
            "hnsw": self._build_hnsw,
            "ivfpq": self._build_ivfpq_trained,
        }

        try:
            build_fn = builders[strategy]
        except KeyError:
            raise ValueError(f"Unknown index strategy: {strategy}")

        return build_fn(X, I)

    # ---------- concrete builders ----------

    def _build_flat(self, X: np.ndarray, I: np.ndarray) -> faiss.Index:
        base = faiss.IndexFlatIP(self.cfg.dim)
        index = faiss.IndexIDMap(base)
        index.add_with_ids(X, I)
        return index

    def _build_hnsw(self, X: np.ndarray, I: np.ndarray) -> faiss.Index:
        base = faiss.IndexHNSWFlat(self.cfg.dim, self.cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = int(self.cfg.hnsw_ef_construction)
        index = faiss.IndexIDMap(base)
        index.add_with_ids(X, I)
        return index

    def _build_ivfpq_trained(self, X: np.ndarray, I: np.ndarray) -> faiss.Index:
        """
        Train IVF-PQ on a subset of X (if needed) and add all vectors.
        """
        N, _ = X.shape
        max_train = min(self.cfg.max_train_vectors, N)

        if N > max_train:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=max_train, replace=False)
            train = X[idx]
        else:
            train = X

        n_train = train.shape[0]
        nlist = self._choose_nlist(n_train)

        quant = faiss.IndexFlatIP(self.cfg.dim)
        ivfpq = faiss.IndexIVFPQ(quant, self.cfg.dim, nlist, self.cfg.pq_m, self.cfg.pq_bits)
        ivfpq.metric_type = faiss.METRIC_INNER_PRODUCT
        ivfpq.verbose = False

        log("index", [
            "IVF-PQ training",
            f"train vectors : {n_train}",
            f"nlist         : {nlist}",
            f"PQ m/bits     : {self.cfg.pq_m} / {self.cfg.pq_bits}",
        ])

        ivfpq.train(train)
        ivfpq.add_with_ids(X, I)
        return ivfpq


# ----------------------------------------------------------------------
# Patch index builder (slices → patches → embeddings → index + manifest)
# ----------------------------------------------------------------------

class PatchIndexBuilder:
    """
    Drives the full pipeline:

      VolumeHelper
        → slices via iter_slices_fibonacci
        → multi-scale, overlapping patches
        → validity filtering
        → batched DINO embeddings
        → FAISS index + manifest

    Public API:
      - build() -> (faiss.Index, pandas.DataFrame)
      - save(index, df, out_dir) -> (index_path, manifest_path)
      - run(out_dir) -> (index_path, manifest_path)
    """

    def __init__(
        self,
        vol_helper: VolumeHelper,
        *,
        k_normals: int = K_NORMALS,
        sampling_cfg: PatchSamplingConfig | None = None,
        index_cfg: IndexConfig | None = None,
    ):
        self.vol = vol_helper
        self.k_normals = int(k_normals)
        self.sampling_cfg = sampling_cfg or PatchSamplingConfig()
        self.index_cfg = index_cfg or IndexConfig()
        self.index_manager = FaissIndexManager(self.index_cfg)

        # Precomputed patch grids per scale:
        #   scale -> List[(patch_row, patch_col, y0, y1, x0, x1)]
        self._patch_grids: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
        self._build_patch_grids()

    # ---------- geometry helpers ----------

    def _build_patch_grids(self) -> None:
        """
        Precompute patch coordinates for each scale once, based on:
          - slice_size_px
          - patch_scales
          - patch_overlap

        For each scale, we store a list of:
          (patch_row, patch_col, y0, y1, x0, x1)
        """
        slice_size = int(self.sampling_cfg.slice_size_px)
        grids: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}

        for scale in self.sampling_cfg.patch_scales:
            s = int(scale)
            window_px, step_px, n_side = self._compute_patch_grid(s)

            coords: List[Tuple[int, int, int, int, int, int]] = []
            for iy in range(n_side):
                y0 = iy * step_px
                y1 = y0 + window_px
                if y1 > slice_size:
                    continue

                for ix in range(n_side):
                    x0 = ix * step_px
                    x1 = x0 + window_px
                    if x1 > slice_size:
                        continue

                    coords.append((iy, ix, y0, y1, x0, x1))

            grids[s] = coords

        self._patch_grids = grids

    def _compute_patch_grid(self, scale: int) -> Tuple[int, int, int]:
        """
        Compute sliding-window parameters for a given scale.

        scale:
          - number of base tiles per side (if there was no overlap)
        Returns:
          (window_px, step_px, n_side)
        """
        slice_px = self.sampling_cfg.slice_size_px
        overlap = self.sampling_cfg.patch_overlap

        if scale <= 0:
            raise ValueError("scale must be > 0")
        if not (0.0 < overlap < 1.0):
            raise ValueError("patch_overlap must be in (0,1)")

        window = int(round(slice_px / float(scale)))
        window = max(1, min(window, slice_px))
        step = max(1, int(round(window * (1.0 - overlap))))

        n_side = int(np.floor((slice_px - window) / step)) + 1
        n_side = max(1, n_side)

        return window, step, n_side

    def _estimate_total_patches(
        self,
        total_slices: int,
    ) -> Tuple[int, Dict[int, int]]:
        """
        Rough estimate of patches per scale and overall.

        Uses precomputed patch grids.
        """
        if not self._patch_grids:
            self._build_patch_grids()

        per_scale: Dict[int, int] = {}
        for s in self.sampling_cfg.patch_scales:
            s_int = int(s)
            coords = self._patch_grids.get(s_int, [])
            per_scale[s_int] = len(coords)

        per_slice = sum(per_scale.values())
        return total_slices * per_slice, per_scale
    
    # ---------- patch validity (slice-level mask + integral) ----------

    def _build_slice_mask_and_integral(
        self,
        slc,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a binary foreground mask and its integral image for a full slice.

        Mask is defined by:
          img > (lo + bg_threshold * (hi - lo))
        where (lo,hi) are global volume intensity bounds.
        """
        cfg = self.sampling_cfg
        lo, hi = self.vol.get_global_intensity_bounds()
        thr = lo + float(cfg.bg_threshold) * (hi - lo)

        img = slc.image
        # Binary mask in {0,1}
        mask = (img > thr).astype(np.uint8)
        integral = mask.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)

        return mask, integral

    @staticmethod
    def _fg_ratio_from_integral(
        integral: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> float:
        """
        Compute foreground ratio for [y0:y1, x0:x1) using an integral image.

        integral[y,x] = sum(mask[0:y, 0:x]) inclusive.
        """
        if x0 >= x1 or y0 >= y1:
            return 0.0

        y1m = y1 - 1
        x1m = x1 - 1

        A = integral[y1m, x1m]
        B = integral[y0 - 1, x1m] if y0 > 0 else 0
        C = integral[y1m, x0 - 1] if x0 > 0 else 0
        D = integral[y0 - 1, x0 - 1] if (x0 > 0 and y0 > 0) else 0

        fg_count = A - B - C + D
        area = (y1 - y0) * (x1 - x0)
        if area <= 0:
            return 0.0
        return float(fg_count) / float(area)

    # ---------- per-slice patch iterator ----------
    
    def _iter_patches_for_slice(
        self,
        slc,
        meta: Dict,
    ) -> Iterator[Tuple[np.ndarray, PatchMeta]]:
        """
        Given a Slice + its pose metadata, yield (patch_img, PatchMeta) pairs
        for all valid patches across all scales.

        Uses:
          - precomputed patch grids per scale
          - slice-level integral image for fast foreground ratio
        """
        img = slc.image
        slice_size = self.sampling_cfg.slice_size_px

        normal_idx = int(meta["normal_idx"])
        depth_idx = int(meta["depth_idx"])
        rot_idx = int(meta["rot_idx"])
        nx, ny, nz = meta["normal_xyz_unit"]
        depth_vox = float(meta["depth_vox"])
        rotation_deg = float(meta["rotation_deg"])

        # Build mask + integral *once* per slice
        _, integral = self._build_slice_mask_and_integral(slc)
        min_fg = float(self.sampling_cfg.min_fg_ratio)

        for scale in self.sampling_cfg.patch_scales:
            s_int = int(scale)
            coords = self._patch_grids.get(s_int, [])
            if not coords:
                continue

            for (iy, ix, y0, y1, x0, x1) in coords:
                # Safety check for unexpected slice size
                if y1 > slice_size or x1 > slice_size:
                    continue

                # Fast foreground ratio via integral image
                fg_ratio = self._fg_ratio_from_integral(integral, x0, y0, x1, y1)
                if fg_ratio < min_fg:
                    continue

                patch = img[y0:y1, x0:x1]

                # center in slice pixel coordinates
                cx_px = (x0 + x1 - 1) / 2.0
                cy_px = (y0 + y1 - 1) / 2.0

                center_xyz = slc.pixel_to_voxel(cx_px, cy_px)
                center_xyz_vox = (
                    float(center_xyz[0]),
                    float(center_xyz[1]),
                    float(center_xyz[2]),
                )

                meta_patch = PatchMeta(
                    normal_idx=normal_idx,
                    depth_idx=depth_idx,
                    rot_idx=rot_idx,
                    normal_x=float(nx),
                    normal_y=float(ny),
                    normal_z=float(nz),
                    depth_vox=depth_vox,
                    rotation_deg=rotation_deg,
                    scale=s_int,
                    patch_row=int(iy),
                    patch_col=int(ix),
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    patch_h=int(y1 - y0),
                    patch_w=int(x1 - x0),
                    slice_size_px=int(slice_size),
                    resolution_um=int(self.sampling_cfg.resolution_um),
                    center_xyz_vox=center_xyz_vox,
                )

                yield patch, meta_patch

    # ---------- batch flush helper ----------

    def _flush_batch(
        self,
        patch_buf: List[np.ndarray],
        meta_buf: List[PatchMeta],
        all_vecs: List[np.ndarray],
        all_ids: List[int],
        rows: List[Dict],
        next_id: int,
    ) -> int:
        """
        Embed all patches in patch_buf as a batch, create rows & IDs, and
        append to all_vecs / all_ids / rows. Returns updated next_id.
        """
        if not patch_buf:
            return next_id

        embs = model.embed_batch(patch_buf)  # (B,D)
        if embs.shape[1] != self.index_cfg.dim:
            raise ValueError(
                f"DINO embed_batch dim {embs.shape[1]} != index_cfg.dim={self.index_cfg.dim}"
            )

        for emb, meta in zip(embs, meta_buf):
            pid = next_id
            next_id += 1

            all_vecs.append(emb.astype(np.float32, copy=False))
            all_ids.append(pid)

            rows.append(
                {
                    "id": pid,
                    # slice pose (3D)
                    "normal_idx": meta.normal_idx,
                    "depth_idx": meta.depth_idx,
                    "rot_idx": meta.rot_idx,
                    "normal_x": meta.normal_x,
                    "normal_y": meta.normal_y,
                    "normal_z": meta.normal_z,
                    "depth_vox": meta.depth_vox,
                    "rotation_deg": meta.rotation_deg,
                    # patch-level info
                    "scale": meta.scale,
                    "patch_row": meta.patch_row,
                    "patch_col": meta.patch_col,
                    "x0": meta.x0,
                    "y0": meta.y0,
                    "x1": meta.x1,
                    "y1": meta.y1,
                    "patch_h": meta.patch_h,
                    "patch_w": meta.patch_w,
                    # global slice info
                    "slice_size_px": meta.slice_size_px,
                    "resolution_um": meta.resolution_um,
                    # patch 3D center
                    "center_x_vox": meta.center_xyz_vox[0],
                    "center_y_vox": meta.center_xyz_vox[1],
                    "center_z_vox": meta.center_xyz_vox[2],
                }
            )

        patch_buf.clear()
        meta_buf.clear()
        return next_id

    # ---------- main build ----------

    def build(self) -> Tuple[faiss.Index, pd.DataFrame, np.ndarray]:
        """
        Run the full pipeline in-memory. Does NOT write to disk.
        """
        slice_size = self.sampling_cfg.slice_size_px

        # Normalize volume once
        if not self.vol.is_normalized():
            log("slices", ["Normalizing volume to [0,1] once..."])
            self.vol.normalize_volume()

        # Get iterator and total_slices
        it, total_slices = iter_slices_fibonacci(
            self.vol,
            k_normals=self.k_normals,
            size_px=slice_size,
            linear_interp=True,
            include_annotation=False,
        )

        est_total_patches, per_scale = self._estimate_total_patches(total_slices)

        log("slices", [
            f"K_NORMALS        : {self.k_normals}",
            f"SLICE_SIZE       : {slice_size}",
            f"PATCH_SCALES     : {self.sampling_cfg.patch_scales}",
            f"PATCH_OVERLAP    : {self.sampling_cfg.patch_overlap}",
            f"Total slices     : {total_slices}",
            "Per-slice patches: "
            + ", ".join(f"s={s} -> {per_scale[s]}" for s in self.sampling_cfg.patch_scales),
            f"Estimated total patches: {est_total_patches}",
        ])

        all_vecs: List[np.ndarray] = []
        all_ids: List[int] = []
        rows: List[Dict] = []
        next_id = 0

        patch_buf: List[np.ndarray] = []
        meta_buf: List[PatchMeta] = []
        batch_size = int(self.sampling_cfg.batch_size)

        bar = tqdm(
            it,
            total=total_slices,
            desc="Extracting patches & embeddings",
            unit="slice",
            dynamic_ncols=True,
        )

        for slc, meta in bar:
            # Fast reject of mostly empty slices
            if not self.vol.is_valid_slice(slc):
                continue

            # Delegate all patch logic to the generator
            for patch, pmeta in self._iter_patches_for_slice(slc, meta):
                patch_buf.append(patch.astype(np.float32, copy=False))
                meta_buf.append(pmeta)

                if len(patch_buf) >= batch_size:
                    next_id = self._flush_batch(
                        patch_buf, meta_buf,
                        all_vecs, all_ids, rows,
                        next_id,
                    )

        # Flush any remaining patches
        next_id = self._flush_batch(
            patch_buf, meta_buf,
            all_vecs, all_ids, rows,
            next_id,
        )

        n_vecs = len(all_vecs)
        log("index", [
            f"Collected patches: {n_vecs}",
            f"Unique ids       : {len(all_ids)}",
        ])

        if n_vecs == 0:
            raise RuntimeError("No valid patches collected; check thresholds / settings.")

        X = np.stack(all_vecs, axis=0).astype(np.float32)  # (N,D)
        I = np.asarray(all_ids, dtype=np.int64)

        index = self.index_manager.build_index(X, I)
        df = pd.DataFrame(rows)
        return index, df, X

    # ---------- save & run helpers ----------

    @staticmethod
    def save(
        index: faiss.Index,
        df: pd.DataFrame,
        vectors: np.ndarray,
        out_dir: Path,
        *,
        index_name: str = "patch_index.faiss",
        manifest_name: str = "patch_manifest.parquet",
        vectors_name: str = "patch_vectors.npy",
    ) -> Tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / index_name
        manifest_path = out_dir / manifest_name
        vectors_path = out_dir / vectors_name

        faiss.write_index(index, str(index_path))
        df.to_parquet(manifest_path, index=False)

        # save embedding matrix
        np.save(vectors_path, vectors.astype(np.float32, copy=False))

        log("index", [
            "Saved index, manifest & vectors",
            f"Index path   : {index_path}",
            f"Manifest     : {manifest_path}",
            f"Vectors      : {vectors_path}",
            f"# rows       : {len(df)}",
            f"Vectors shape: {vectors.shape}",
        ])
        return index_path, manifest_path

    def run(
        self,
        out_dir: Path,
        *,
        index_name: str = "patch_index.faiss",
        manifest_name: str = "patch_manifest.parquet",
    ) -> Tuple[Path, Path]:
        """
        Full pipeline: build in memory, then write to disk.
        """
        index, df, X = self.build()
        return self.save(
            index,
            df,
            X,
            out_dir,
            index_name=index_name,
            manifest_name=manifest_name,
        )