# index/store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from .config import OUT_DIR

__all__ = [
    "IndexStore",
]


class IndexStore:
    """
    Simple storage layer for the *patch-based* index.

    - Loads:
        - patch_index.faiss       (FAISS index with all patch embeddings)
        - patch_manifest.parquet  (one row per patch, id as index)
        - patch_vectors.npy       (optional: full (N,D) embedding matrix)

    - Provides:
        - .index      -> faiss.Index
        - .manifest   -> pd.DataFrame (indexed by patch id)
        - .search(Q,k)-> (D, I) with cosine / IP
        - helpers to fetch metadata for specific ids
        - vectors_for_ids(ids) -> (len(ids), D) np.ndarray
    """

    def __init__(
        self,
        root: Path | str = OUT_DIR,
        *,
        index_name: str = "patch_index.faiss",
        manifest_name: str = "patch_manifest.parquet",
        vectors_name: str = "patch_vectors.npy",
    ):
        self.root = Path(root)
        self.index_name = index_name
        self.manifest_name = manifest_name
        self.vectors_name = vectors_name

        self._index: Optional[faiss.Index] = None
        self._manifest_df: Optional[pd.DataFrame] = None
        self._vectors: Optional[np.ndarray] = None  # (N,D) or None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_all(self) -> "IndexStore":
        """
        Load FAISS index + manifest (and optionally vectors) from disk.
        """
        index_path = self.root / self.index_name
        manifest_path = self.root / self.manifest_name
        vectors_path = self.root / self.vectors_name

        self._index = self._load_faiss(index_path)
        df = self._read_parquet(manifest_path)

        # Use "id" as index if present
        if "id" in df.columns:
            df = df.set_index("id")

        self._manifest_df = df

        # Optional: load vectors sidecar if present
        if vectors_path.exists():
            self._vectors = np.load(vectors_path, mmap_mode="r")
        else:
            self._vectors = None

        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load_all() first.")
        return self._index

    @property
    def manifest(self) -> pd.DataFrame:
        if self._manifest_df is None:
            raise RuntimeError("Manifest not loaded. Call load_all() first.")
        return self._manifest_df

    @property
    def dim(self) -> int:
        """Embedding dimension (D)."""
        return int(self.index.d)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return int(self.index.ntotal)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_queries(Q: np.ndarray) -> np.ndarray:
        if Q.dtype != np.float32:
            Q = Q.astype(np.float32)
        faiss.normalize_L2(Q)
        return Q

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity search (via inner product on L2-normalized rows).

        Parameters
        ----------
        Q : np.ndarray
            Query vectors, shape (nq, D) or (D,)
        k : int
            Number of neighbours.

        Returns
        -------
        D, I : np.ndarray, np.ndarray
            Distances and ids from FAISS, both shape (nq, k).
        """
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Qn = np.asarray(Q, dtype=np.float32).copy()   # copy to avoid in-place mutation
        Qn = self._normalize_queries(Q)
        D, I = self.index.search(Qn, k)
        return D, I

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def rows_for_ids(self, ids: Iterable[int]) -> pd.DataFrame:
        """
        Return manifest rows corresponding to given patch ids.
        """
        df = self.manifest
        ids_list = [int(i) for i in ids]
        # .reindex keeps order of ids_list, even if some are missing
        return df.reindex(ids_list)

    def row_for_id(self, pid: int) -> pd.Series:
        """
        Return a single manifest row for a given patch id.
        """
        df = self.manifest
        return df.loc[int(pid)]

    def pose_for_id(self, pid: int) -> Dict:
        """
        Convenience: extract 3D pose + patch box for a given patch id.
        """
        row = self.row_for_id(pid)

        keys = [
            "normal_idx",
            "depth_idx",
            "rot_idx",
            "normal_x",
            "normal_y",
            "normal_z",
            "depth_vox",
            "rotation_deg",
            "scale",
            "patch_row",
            "patch_col",
            "x0",
            "y0",
            "x1",
            "y1",
            "patch_h",
            "patch_w",
            "slice_size_px",
            "resolution_um",
            "center_x_vox",
            "center_y_vox",
            "center_z_vox",
        ]

        out = {}
        for k in keys:
            if k in row:
                out[k] = row[k]
        return out

    # ------------------------------------------------------------------
    # Vector helpers for reranker
    # ------------------------------------------------------------------

    def vectors_for_ids(self, ids: Iterable[int]) -> np.ndarray:
        """
        Return embedding vectors for given patch ids.

        Preferred path:
          - if a sidecar matrix (patch_vectors.npy) is loaded, index into it.
        """
        ids_list = [int(i) for i in ids]

        if self._vectors is not None:
            vecs = np.asarray(self._vectors, dtype=np.float32)
            return vecs[ids_list]

        raise RuntimeError(
            "patch_vectors.npy not loaded; cannot fetch candidate embeddings for reranker. "
            "Regenerate patch_vectors.npy or load it."
        )

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_faiss(path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {path}")
        return faiss.read_index(str(path))

    @staticmethod
    def save_faiss(index: faiss.Index, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet manifest: {path}")
        return pd.read_parquet(path)
