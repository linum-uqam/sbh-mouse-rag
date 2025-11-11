from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional

import faiss
import numpy as np
import pandas as pd

from .config import OUT_DIR, HNSW_M, HNSW_EF_CONSTRUCTION, SLICE_SCALES, TOKEN_INDEX_PATH

__all__ = [
    "IndexStore",
    "build_flat_ip",
    "build_hnsw_ip",
    "wrap_with_ids",
    "write_manifest_slice",
    "pack_token_id",
    "unpack_token_id",
]

# ---- FAISS builders ----
def build_flat_ip(d: int) -> faiss.Index:
    return faiss.IndexFlatIP(d)

def build_hnsw_ip(d: int) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    return idx

def wrap_with_ids(index: faiss.Index) -> faiss.Index:
    return faiss.IndexIDMap2(index)

# ---- Manifest writer ----
def write_manifest_slice(rows: List[Dict], path: Union[str, Path]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)

# ---- Token id helpers (slice_id << 16) | token_idx ----
def pack_token_id(slice_id: int, token_idx: int) -> int:
    return (int(slice_id) << 16) | int(token_idx & 0xFFFF)

def unpack_token_id(packed: int) -> Tuple[int, int]:
    return (int(packed) >> 16), int(packed & 0xFFFF)

# ---- IndexStore ----
class IndexStore:
    def __init__(self, root: Path | str = OUT_DIR):
        self.root = Path(root)

        self._coarse: faiss.Index | None = None
        self._coarse_by_scale: Dict[int, faiss.Index] = {}

        self._token_ivfpq: Optional[faiss.Index] = None

        self._manifest_df: pd.DataFrame | None = None
        self._id_to_tokenpath: Dict[int, str] | None = None
        self._id_to_maskpath: Dict[int, str] | None = None

    # ---------- Loaders ----------
    def load_all(self) -> "IndexStore":
        self._coarse = self._load_faiss(self.root / "coarse.faiss")

        # Per-scale indices are mandatory in your new build
        self._coarse_by_scale = {}
        for k in SLICE_SCALES:
            p = self.root / f"coarse.scale{k}.faiss"
            self._coarse_by_scale[k] = self._load_faiss(p)

        # Token IVF-PQ (optional but expected)
        tip = TOKEN_INDEX_PATH
        if tip.exists():
            self._token_ivfpq = self._load_faiss(tip)

        self._manifest_df = self._read_parquet(self.root / "manifest.parquet").set_index("id")
        self._id_to_tokenpath = self._build_id_to_tokenpath(self._manifest_df)
        if "mask_path" in self._manifest_df.columns:
            self._id_to_maskpath = dict(zip(self._manifest_df.index.astype(int).tolist(),
                                            self._manifest_df["mask_path"].astype(str).tolist()))
        else:
            self._id_to_maskpath = {}
        return self

    # ---------- Accessors ----------
    @property
    def coarse(self) -> faiss.Index:
        assert self._coarse is not None, "IndexStore not loaded. Call load_all()."
        return self._coarse

    def coarse_scale(self, k: int) -> faiss.Index:
        assert k in self._coarse_by_scale, f"Missing per-scale index for k={k}"
        return self._coarse_by_scale[k]

    @property
    def token_ivfpq(self) -> Optional[faiss.Index]:
        return self._token_ivfpq

    @property
    def manifest(self) -> pd.DataFrame:
        assert self._manifest_df is not None, "IndexStore not loaded. Call load_all()."
        return self._manifest_df

    def token_path(self, slice_id: int) -> str | None:
        assert self._id_to_tokenpath is not None, "IndexStore not loaded. Call load_all()."
        return self._id_to_tokenpath.get(int(slice_id))

    def mask_path(self, slice_id: int) -> str | None:
        assert self._id_to_maskpath is not None, "IndexStore not loaded. Call load_all()."
        return self._id_to_maskpath.get(int(slice_id))

    # ---------- Search helpers ----------
    @staticmethod
    def _l2norm_rows(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-12)

    def search(self, index: faiss.Index, Q: np.ndarray, k: int):
        if Q.dtype != np.float32:
            Q = Q.astype(np.float32)
        faiss.normalize_L2(Q)
        return index.search(Q, k)

    def search_tokens(self, Q: np.ndarray, topM: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Q: (nq, D) float32 (should be L2-normalized rows)
        Returns (D, I) from FAISS: shapes (nq, topM)
        """
        assert self._token_ivfpq is not None, "Token IVF-PQ not loaded."
        if Q.dtype != np.float32:
            Q = Q.astype(np.float32)
        faiss.normalize_L2(Q)
        return self._token_ivfpq.search(Q, topM)

    # ---------- I/O ----------
    @staticmethod
    def _load_faiss(path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(f"Missing index: {path}")
        return faiss.read_index(str(path))
    
    @staticmethod
    def save_faiss(index: faiss.Index, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet: {path}")
        return pd.read_parquet(path)

    @staticmethod
    def _build_id_to_tokenpath(manifest_df: pd.DataFrame) -> Dict[int, str]:
        ids = manifest_df.index.astype(int)
        paths = manifest_df["token_path"].astype(str)
        return dict(zip(ids.tolist(), paths.tolist()))
