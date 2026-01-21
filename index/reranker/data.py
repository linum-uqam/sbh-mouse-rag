from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from index.utils import load_image_gray  # keep your existing helper
from index.model.dino import model as dino_model

from .config import TrainingConfig


# -----------------------------
# IO helpers
# -----------------------------

def _log(msg: str) -> None:
    print(msg)


def load_hits_csv(cfg: TrainingConfig) -> pd.DataFrame:
    path = Path(cfg.hits_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing hits CSV: {path}")
    df = pd.read_csv(path)

    required = [cfg.col_row_idx, cfg.col_source, cfg.col_patch_id, cfg.col_rank]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"hits CSV missing {missing}. Present: {list(df.columns)}")

    if cfg.col_gt_prob not in df.columns and cfg.col_gt_logit not in df.columns:
        raise ValueError(
            f"hits CSV must contain '{cfg.col_gt_prob}' or '{cfg.col_gt_logit}'. "
            f"Present: {list(df.columns)}"
        )

    _log(f"[Reranker] Loaded hits CSV: path={path} rows={len(df)}")
    return df


def load_dataset_csv(cfg: TrainingConfig) -> pd.DataFrame:
    path = Path(cfg.dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {path}")
    df = pd.read_csv(path)

    required = ["allen_path", "real_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dataset CSV missing {missing}. Present: {list(df.columns)}")

    _log(f"[Reranker] Loaded dataset CSV: path={path} rows={len(df)}")
    return df


def load_patch_vectors(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing patch_vectors.npy: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"patch_vectors must be (N,D), got {arr.shape}")
    _log(f"[Reranker] Loaded patch vectors: path={path} shape={arr.shape} dtype={arr.dtype}")
    return arr


def load_patch_id_mapping(manifest_path: str | Path, n_vectors: int) -> Optional[Dict[int, int]]:
    """
    Return mapping patch_id -> row_index_in_patch_vectors.
    If patch_id is already 0..N-1 aligned, returns None (identity).
    """
    mp = Path(manifest_path)
    if not mp.exists():
        _log(f"[Reranker] patch_manifest not found at {mp}; assuming patch_id indexes vectors directly.")
        return None

    df = pd.read_parquet(mp)
    if "patch_id" not in df.columns:
        _log("[Reranker] patch_manifest has no 'patch_id'; assuming identity mapping.")
        return None

    patch_ids = df["patch_id"].to_numpy(dtype=np.int64)
    if len(patch_ids) != n_vectors:
        _log(
            "[Reranker] patch_manifest row count != n_vectors; building explicit mapping.\n"
            f"  - n_manifest: {len(patch_ids)}\n"
            f"  - n_vectors : {n_vectors}"
        )
        return {int(pid): int(i) for i, pid in enumerate(patch_ids)}

    if np.array_equal(patch_ids, np.arange(n_vectors, dtype=np.int64)):
        _log("[Reranker] Detected identity patch_id mapping.")
        return None

    _log("[Reranker] Building patch_id -> vector_row mapping from manifest.")
    return {int(pid): int(i) for i, pid in enumerate(patch_ids)}


# -----------------------------
# Query embedding cache
# -----------------------------

def _resolve_query_image_path(cfg: TrainingConfig, dataset_df: pd.DataFrame, row_idx: int, source: str) -> Path:
    if row_idx < 0 or row_idx >= len(dataset_df):
        raise IndexError(f"row_idx {row_idx} out of bounds (dataset rows={len(dataset_df)})")

    ds_row = dataset_df.iloc[row_idx]
    if source == "allen":
        p = Path(str(ds_row["allen_path"]))
    elif source == "real":
        p = Path(str(ds_row["real_path"]))
    else:
        raise ValueError(f"Unknown source={source!r}")

    if not p.exists():
        candidate = Path(cfg.dataset_csv).parent / p
        if candidate.exists():
            p = candidate

    if not p.exists():
        raise FileNotFoundError(f"Query image not found: {p}")
    return p


def _embed_global_batch(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Wrapper around your DINO helper.
    Expected output: (B, D)
    """
    if hasattr(dino_model, "embed_global_batch"):
        out = dino_model.embed_global_batch(imgs)
    elif hasattr(dino_model, "embed_batch"):
        out = dino_model.embed_batch(imgs)
    else:
        tok = dino_model.embed_tokens_batch(imgs, pool=1)  # (B,T,D)
        out = tok.mean(axis=1)

    out = np.asarray(out)
    if out.ndim != 2:
        raise ValueError(f"Expected (B,D), got {out.shape}")
    return out


def build_or_load_query_vectors(
    cfg: TrainingConfig,
    hits_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    *,
    embed_dim: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict[Tuple[int, str], int]]:
    """
    Returns:
      q_vecs: (n_queries, D) float32
      key_to_qidx: mapping (row_idx, source) -> qidx
    """
    keys_df = hits_df[[cfg.col_row_idx, cfg.col_source]].drop_duplicates().reset_index(drop=True)
    n_queries = len(keys_df)
    cache_path = Path(cfg.query_vectors_cache)

    key_to_qidx: Dict[Tuple[int, str], int] = {
        (int(r[cfg.col_row_idx]), str(r[cfg.col_source])): int(i)
        for i, r in keys_df.iterrows()
    }

    if cache_path.exists():
        q = np.load(cache_path)
        if q.shape == (n_queries, embed_dim):
            _log(f"[Reranker] Loaded query vectors cache: path={cache_path} shape={q.shape}")
            return q.astype(np.float32, copy=False), key_to_qidx
        _log(f"[Reranker] Query cache mismatch; rebuilding. cache_shape={q.shape} expected={(n_queries, embed_dim)}")

    _log(f"[Reranker] Building query vectors: n_queries={n_queries}")

    q_vecs = np.zeros((n_queries, embed_dim), dtype=np.float32)
    batch_imgs: List[np.ndarray] = []
    batch_qidx: List[int] = []

    for qidx, row in keys_df.iterrows():
        row_idx = int(row[cfg.col_row_idx])
        src = str(row[cfg.col_source])
        img_path = _resolve_query_image_path(cfg, dataset_df, row_idx, src)
        img = load_image_gray(img_path)  # float [0,1]
        batch_imgs.append(img)
        batch_qidx.append(int(qidx))

        if len(batch_imgs) >= batch_size:
            emb = _embed_global_batch(batch_imgs).astype(np.float32, copy=False)
            if emb.shape[1] != embed_dim:
                raise ValueError(f"Embedding dim {emb.shape[1]} != expected {embed_dim}")
            for i, qi in enumerate(batch_qidx):
                q_vecs[qi] = emb[i]
            batch_imgs.clear()
            batch_qidx.clear()

    if batch_imgs:
        emb = _embed_global_batch(batch_imgs).astype(np.float32, copy=False)
        if emb.shape[1] != embed_dim:
            raise ValueError(f"Embedding dim {emb.shape[1]} != expected {embed_dim}")
        for i, qi in enumerate(batch_qidx):
            q_vecs[qi] = emb[i]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, q_vecs)
    _log(f"[Reranker] Saved query vectors cache: path={cache_path} shape={q_vecs.shape}")

    return q_vecs, key_to_qidx


# -----------------------------
# Build fixed-size lists + targets
# -----------------------------

def _stable_softmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    x: (K,) float
    mask: (K,) bool
    returns p: (K,) with p.sum()=1 over mask (or uniform if degenerate)
    """
    p = np.zeros_like(x, dtype=np.float32)
    if not np.any(mask):
        return p

    xf = x[mask].astype(np.float64)
    m = float(np.max(xf))
    ex = np.exp(xf - m)
    Z = float(ex.sum())
    if Z <= 0 or not np.isfinite(Z):
        p[mask] = 1.0 / float(mask.sum())
        return p

    p[mask] = (ex / Z).astype(np.float32)
    return p


def build_lists_and_targets(
    cfg: TrainingConfig,
    hits_df: pd.DataFrame,
    key_to_qidx: Dict[Tuple[int, str], int],
    patch_id_to_vecrow: Optional[Dict[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cand_vec_rows: (n_queries, K) int64  (row indices into patch_vectors)
      gt_prob:       (n_queries, K) float32 (soft labels; 0 on invalid)
      mask:          (n_queries, K) bool
      row_idx_of_q:  (n_queries,) int64 (for splitting)
      kept_qidx:     (n_kept,) int64 (queries kept for training)
    """
    K = int(cfg.list_k)
    n_queries = len(key_to_qidx)

    cand_vec_rows = np.full((n_queries, K), -1, dtype=np.int64)
    gt_prob = np.zeros((n_queries, K), dtype=np.float32)
    mask = np.zeros((n_queries, K), dtype=bool)
    row_idx_of_q = np.full((n_queries,), -1, dtype=np.int64)

    # Filter ranks early
    df = hits_df.copy()
    df[cfg.col_rank] = df[cfg.col_rank].astype(np.int64)
    df = df[df[cfg.col_rank] <= int(cfg.train_topk)]

    # Group indices by query key
    grouped: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for i, r in df.iterrows():
        key = (int(r[cfg.col_row_idx]), str(r[cfg.col_source]))
        grouped[key].append(int(i))

    use_prob = (cfg.col_gt_prob in df.columns)
    use_logit = (cfg.col_gt_logit in df.columns)

    if not use_prob and not use_logit:
        raise ValueError("Need gt_prob or gt_logit in hits CSV.")

    for (row_idx, src), qidx in key_to_qidx.items():
        row_idx_of_q[qidx] = int(row_idx)

        idxs = grouped.get((int(row_idx), str(src)), [])
        if not idxs:
            continue

        d = df.loc[idxs, [cfg.col_patch_id, cfg.col_rank] + ([cfg.col_gt_prob] if use_prob else []) + ([cfg.col_gt_logit] if use_logit else [])].copy()
        d[cfg.col_patch_id] = d[cfg.col_patch_id].astype(np.int64)
        d[cfg.col_rank] = d[cfg.col_rank].astype(np.int64)

        # Deduplicate by patch_id: keep best rank (lowest)
        d.sort_values(cfg.col_rank, ascending=True, inplace=True)
        d = d.drop_duplicates(subset=[cfg.col_patch_id], keep="first")

        # Take first K
        d = d.head(K)

        pids = d[cfg.col_patch_id].to_numpy(dtype=np.int64)

        # Map patch_id -> vector row
        if patch_id_to_vecrow is None:
            vec_rows = pids
        else:
            vec_rows = np.asarray([patch_id_to_vecrow.get(int(pid), -1) for pid in pids], dtype=np.int64)

        ok = vec_rows >= 0
        vec_rows = vec_rows[ok]

        if vec_rows.size == 0:
            continue

        n = int(min(K, vec_rows.size))
        vec_rows = vec_rows[:n]
        valid_mask = np.zeros((K,), dtype=bool)
        valid_mask[:n] = True

        # Build target probs for these n entries
        if use_prob:
            p = d[cfg.col_gt_prob].to_numpy(dtype=np.float32)[ok][:n]
            # If user truncates, renormalize within kept list
            p = np.where(np.isfinite(p), p, 0.0).astype(np.float32, copy=False)
            s = float(p.sum())
            if s <= 0:
                p = np.full((n,), 1.0 / float(n), dtype=np.float32)
            else:
                p = p / s
        else:
            # Fallback: use gt_logit and re-softmax
            logits = d[cfg.col_gt_logit].to_numpy(dtype=np.float32)[ok][:n]
            logits = np.where(np.isfinite(logits), logits, -1e9).astype(np.float32, copy=False)
            p_full = _stable_softmax(logits, np.ones((n,), dtype=bool))
            p = p_full[:n]

        # Store
        cand_vec_rows[qidx, :n] = vec_rows
        mask[qidx, :n] = True
        gt_prob[qidx, :n] = p

    valid_counts = mask.sum(axis=1)
    kept_qidx = np.where(valid_counts >= int(cfg.require_min_candidates))[0].astype(np.int64)

    dropped = int((valid_counts < int(cfg.require_min_candidates)).sum())
    if dropped:
        _log(f"[Reranker] Dropping {dropped} queries with <{cfg.require_min_candidates} valid candidates.")

    _log(f"[Reranker] Kept queries: {len(kept_qidx)} / {n_queries}")
    return cand_vec_rows, gt_prob, mask, row_idx_of_q, kept_qidx


# -----------------------------
# Dataset + splits
# -----------------------------

class ListwiseDataset(Dataset):
    """
    One item = one query with its candidate list.

    Returns:
      q_emb: (D,)
      c_emb: (K,D)
      gt_prob: (K,)   (soft labels; 0 on invalid)
      mask: (K,) bool
    """

    def __init__(
        self,
        q_vecs: np.ndarray,             # (Nq,D)
        patch_vecs: np.ndarray,         # (Np,D) memmap ok
        cand_vec_rows: np.ndarray,      # (Nq,K)
        gt_prob: np.ndarray,            # (Nq,K)
        mask: np.ndarray,               # (Nq,K)
        q_indices: np.ndarray,          # (N_used,)
    ):
        self.q_vecs = q_vecs
        self.patch_vecs = patch_vecs
        self.cand_vec_rows = cand_vec_rows
        self.gt_prob = gt_prob
        self.mask = mask
        self.q_indices = q_indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.q_indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        qi = int(self.q_indices[i])
        q = self.q_vecs[qi]  # (D,)

        rows = self.cand_vec_rows[qi]  # (K,)
        m = self.mask[qi]              # (K,)

        # Safe indexing: invalid -> 0 (will be masked anyway)
        safe_rows = np.where(m, rows, 0).astype(np.int64, copy=False)
        c = np.asarray(self.patch_vecs[safe_rows], dtype=np.float32)  # (K,D)

        p = np.asarray(self.gt_prob[qi], dtype=np.float32)            # (K,)
        # Ensure exact normalization over valid entries (defensive)
        psum = float(p[m].sum()) if np.any(m) else 0.0
        if psum > 0:
            p = p / psum

        return {
            "q_emb": torch.from_numpy(np.asarray(q, dtype=np.float32)),
            "c_emb": torch.from_numpy(c),
            "gt_prob": torch.from_numpy(p),
            "mask": torch.from_numpy(m.astype(np.bool_)),
        }


def _split_by_row_idx(cfg: TrainingConfig, row_idx_of_q: np.ndarray) -> Tuple[Set[int], Set[int], Set[int]]:
    rng = np.random.RandomState(cfg.seed)
    unique_rows = np.unique(row_idx_of_q)
    rng.shuffle(unique_rows)

    n = len(unique_rows)
    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))

    train_rows = set(unique_rows[:n_train].tolist())
    val_rows = set(unique_rows[n_train:n_train + n_val].tolist())
    test_rows = set(unique_rows[n_train + n_val:].tolist())

    _log(
        "[Reranker] Split by row_idx:\n"
        f"  - total row_idx : {n}\n"
        f"  - train         : {len(train_rows)}\n"
        f"  - val           : {len(val_rows)}\n"
        f"  - test          : {len(test_rows)}"
    )
    return train_rows, val_rows, test_rows


def prepare_dataloaders(
    cfg: TrainingConfig,
    *,
    embed_dim: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    hits_df = load_hits_csv(cfg)
    dataset_df = load_dataset_csv(cfg)

    patch_vecs = load_patch_vectors(cfg.patch_vectors_path)
    patch_id_to_vecrow = load_patch_id_mapping(cfg.patch_manifest_path, n_vectors=int(patch_vecs.shape[0]))

    q_vecs, key_to_qidx = build_or_load_query_vectors(cfg, hits_df, dataset_df, embed_dim=embed_dim)

    cand_vec_rows, gt_prob, mask, row_idx_of_q, kept_qidx = build_lists_and_targets(
        cfg, hits_df, key_to_qidx, patch_id_to_vecrow
    )

    train_rows, val_rows, test_rows = _split_by_row_idx(cfg, row_idx_of_q[kept_qidx])

    kept_rows = row_idx_of_q[kept_qidx]
    train_q = kept_qidx[np.isin(kept_rows, np.fromiter(train_rows, dtype=np.int64))]
    val_q   = kept_qidx[np.isin(kept_rows, np.fromiter(val_rows, dtype=np.int64))]
    test_q  = kept_qidx[np.isin(kept_rows, np.fromiter(test_rows, dtype=np.int64))]

    train_ds = ListwiseDataset(q_vecs, patch_vecs, cand_vec_rows, gt_prob, mask, train_q)
    val_ds   = ListwiseDataset(q_vecs, patch_vecs, cand_vec_rows, gt_prob, mask, val_q)
    test_ds  = ListwiseDataset(q_vecs, patch_vecs, cand_vec_rows, gt_prob, mask, test_q)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    _log(
        "[Reranker] Dataset sizes:\n"
        f"  - train queries: {len(train_ds)}\n"
        f"  - val queries  : {len(val_ds)}\n"
        f"  - test queries : {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader
