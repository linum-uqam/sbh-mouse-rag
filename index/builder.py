from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import D, PQ_M, PQ_BITS, TOKEN_INDEX_PATH
from .store import pack_token_id, build_flat_ip, wrap_with_ids, pack_token_id
from .utils import log, l2norm_rows

# ---------- small utilities ----------

def adaptive_pool_tokens_to_vec(tokens: np.ndarray, grid_hw: int, k: int) -> np.ndarray:
    T, Ddim = tokens.shape
    g = tokens.reshape(grid_hw, grid_hw, Ddim)
    rows = np.array_split(g, k, axis=0)
    pooled_cells = []
    for r in rows:
        cols = np.array_split(r, k, axis=1)
        for cell in cols:
            pooled_cells.append(cell.reshape(-1, Ddim).mean(axis=0))
    V = np.stack(pooled_cells, axis=0)   # (k*k, D)
    v = V.mean(axis=0)                   # (D,)
    return v.astype(np.float32, copy=False)

def build_base_index(d: int, use_hnsw: bool, hnsw_m: int, hnsw_ef: int) -> faiss.Index:
    if use_hnsw:
        idx = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = hnsw_ef
        return wrap_with_ids(idx)
    return wrap_with_ids(build_flat_ip(d))

# ---------- builder ----------

class IVFPQ:
    """
    Normative IVF-PQ builder:
      - foreground-only sampling per slice at a fixed ratio
      - data-driven nlist = floor_multiple(max(256, n_train // 40), 128) clamped to 65536
      - IP (cosine) metric throughout
    Assumes df has columns: ['id', 'token_path', 'mask_path'] with valid paths/shapes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        d: int = D,
        m: int = PQ_M,
        nbits: int = PQ_BITS,
        train_ratio: float = 0.5,
        rng_seed: int = 42,
    ):
        self.df = df
        self.d = int(d)
        self.m = int(m)
        self.nbits = int(nbits)
        self.train_ratio = float(train_ratio)
        self.rng = np.random.default_rng(rng_seed)

    @staticmethod
    def _load_tokens_and_fg(row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tok = np.load(row["token_path"], mmap_mode="r").astype(np.float32)
        msk = np.load(row["mask_path"],  mmap_mode="r").astype(bool)
        T = tok.shape[0]
        fg_idx = np.nonzero(msk[:T])[0]
        return tok, msk, fg_idx

    def build_training_set(self) -> Tuple[np.ndarray, int, int]:
        """
        Returns (train_mat_l2, n_train, nlist).
        """
        buf: List[np.ndarray] = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc="Token train collect", dynamic_ncols=True):
            tok, _, fg_idx = self._load_tokens_and_fg(row)
            if fg_idx.size == 0:
                continue
            n_target = int(round(fg_idx.size * self.train_ratio))
            n_target = max(1, min(n_target, fg_idx.size))
            pick = self.rng.choice(fg_idx, size=n_target, replace=False)
            buf.append(tok[pick])

        train_set = l2norm_rows(np.concatenate(buf, axis=0))
        n_train = int(train_set.shape[0])
        nlist = self._choose_ncentroid(n_train)
        return train_set, n_train, nlist

    @staticmethod
    def _choose_ncentroid(n_train: int) -> int:
        raw = n_train // 40
        n = max(256, raw)
        nlist = (n // 128) * 128
        return min(nlist, 65536)

    @staticmethod
    def build_index(d: int, nlist: int, m: int, nbits: int) -> faiss.IndexIVFPQ:
        """
        Construct an untrained IVF-PQ (FAISS verbose off).
        """
        quant = build_flat_ip(d)
        ivfpq = faiss.IndexIVFPQ(quant, d, int(nlist), int(m), int(nbits))
        ivfpq.metric_type = faiss.METRIC_INNER_PRODUCT
        ivfpq.verbose = False # Faiss debug output. 
        return ivfpq

    def train(self) -> Tuple[faiss.IndexIVFPQ, int, int]:
        train_mat = self._build_train_set()
        n_train = int(train_mat.shape[0])
        nlist = self._choose_ncentroid(n_train)

        index = self._build_ivfpq(self.d, nlist, self.m, self.nbits)
        index.train(train_mat)
        return index, n_train, nlist
    
    def add_tokens(self, index: faiss.IndexIVFPQ, flush_every: int = 1_000_000) -> int:
        """
        Add all foreground tokens to a trained IVF-PQ.
        Returns total vectors added.
        """
        id_buf: List[np.ndarray] = []
        vec_buf: List[np.ndarray] = []
        cached = 0
        total_added = 0

        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc="Token add", dynamic_ncols=True):
            sid = int(row["id"])
            tok, _, fg_idx = self._load_tokens_and_fg(row)
            if fg_idx.size == 0:
                continue

            X = l2norm_rows(tok[fg_idx])
            I = np.array([pack_token_id(sid, int(ti)) for ti in fg_idx], dtype=np.int64)

            vec_buf.append(X)
            id_buf.append(I)
            cached += X.shape[0]

            if cached >= flush_every:
                Xt = np.concatenate(vec_buf, axis=0)
                It = np.concatenate(id_buf, axis=0)
                index.add_with_ids(Xt, It)
                total_added += Xt.shape[0]
                vec_buf.clear(); id_buf.clear(); cached = 0

        if vec_buf:
            Xt = np.concatenate(vec_buf, axis=0)
            It = np.concatenate(id_buf, axis=0)
            index.add_with_ids(Xt, It)
            total_added += Xt.shape[0]

        return total_added


# ---------- top-level function ----------

def create_ivfpq(
    manifest_path: Path,
    out_path: Path = TOKEN_INDEX_PATH,
    d: int = D,
    m: int = PQ_M,
    nbits: int = PQ_BITS,
    train_ratio: float = 0.5,
) -> None:
    df = pd.read_parquet(manifest_path)
    ivfpq = IVFPQ(df, d=d, m=m, nbits=nbits, train_ratio=train_ratio)

    # Stage 1: build training matrix & decide nlist
    training_set, n_train, nlist = ivfpq.build_training_set()
    log("tokens", [
        "config",
        f"train vectors : {n_train}",
        f"nlist         : {nlist}",
        f"PQ m/bits     : {m} / {nbits}",
        f"dim (D)       : {d}",
        "metric        : Inner Product (cosine w/ L2 rows)",
    ])
    log("tokens", ["training..."])

    # Stage 1: train
    index = ivfpq.build_index(d, nlist, m, nbits)
    index.train(training_set)

    # Stage 2: add foreground tokens
    log("tokens", ["adding..."])
    total_added = ivfpq.add_tokens(index)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    log("tokens", [
        "done",
        f"added        : {total_added}",
        f"saved        : {out_path}",
        f"summary      : train={n_train}, nlist={nlist}",
    ])
