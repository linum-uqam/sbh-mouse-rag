# index/store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Union

import faiss
import numpy as np
import pandas as pd

from .config import OUT_DIR, HNSW_M, HNSW_EF_CONSTRUCTION

__all__ = [
    "IndexStore",
    "build_flat_ip",
    "build_hnsw_ip",
    "write_manifest_slice",
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

# ---- Manifest writers  ----
def write_manifest_slice(rows: List[Dict], path: Union[str, Path]) -> None:
    """Persist slice-level manifest (id -> token_path, meta...)"""
    pd.DataFrame(rows).to_parquet(path, index=False)

# ---- IndexStore ----
class IndexStore:
    def __init__(self, root: Path | str = OUT_DIR):
        self.root = Path(root)
        self._coarse: faiss.Index | None = None
        self._manifest_df: pd.DataFrame | None = None
        self._id_to_tokenpath: Dict[int, str] | None = None
        self._id_to_maskpath: Dict[int, str] | None = None

    def load_all(self) -> "IndexStore":
        self._coarse = self._load_faiss(self.root / "coarse.faiss")
        self._manifest_df = self._read_parquet(self.root / "manifest.parquet").set_index("id")
        self._id_to_tokenpath = self._build_id_to_tokenpath(self._manifest_df)
        if "mask_path" in self._manifest_df.columns:
            self._id_to_maskpath = dict(zip(self._manifest_df.index.astype(int).tolist(),
                                            self._manifest_df["mask_path"].astype(str).tolist()))
        else:
            self._id_to_maskpath = {}
        return self

    def mask_path(self, slice_id: int) -> str | None:
        assert self._id_to_maskpath is not None, "IndexStore not loaded. Call load_all()."
        return self._id_to_maskpath.get(int(slice_id))

    @property
    def coarse(self) -> faiss.Index:
        assert self._coarse is not None, "IndexStore not loaded. Call load_all()."
        return self._coarse

    @property
    def manifest(self) -> pd.DataFrame:
        assert self._manifest_df is not None, "IndexStore not loaded. Call load_all()."
        return self._manifest_df

    def token_path(self, slice_id: int) -> str | None:
        assert self._id_to_tokenpath is not None, "IndexStore not loaded. Call load_all()."
        return self._id_to_tokenpath.get(int(slice_id))

    def search(self, index: faiss.Index, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if Q.dtype != np.float32:
            Q = Q.astype(np.float32)
        faiss.normalize_L2(Q)
        return index.search(Q, k)

    @staticmethod
    def _load_faiss(path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(f"Missing index: {path}")
        return faiss.read_index(str(path))

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet: {path}")
        return pd.read_parquet(path)

    @staticmethod
    def _build_id_to_tokenpath(manifest_df: pd.DataFrame) -> Dict[int, str]:
        # Map: slice_id (index) -> token_path
        ids = manifest_df.index.astype(int)
        paths = manifest_df["token_path"].astype(str)
        return dict(zip(ids.tolist(), paths.tolist()))
