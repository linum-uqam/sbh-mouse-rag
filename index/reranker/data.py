from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from index.utils import load_image_gray
from index.model.dino import model as dino_model

from .config import TrainingConfig


def _log(msg: str) -> None:
    print(msg)


def load_hits_csv(cfg: TrainingConfig) -> pd.DataFrame:
    path = Path(cfg.hits_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing hits CSV: {path}")
    df = pd.read_csv(path)

    required = [cfg.col_row_idx, cfg.col_source, cfg.col_patch_id, cfg.col_rank, cfg.col_geom_dist]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"hits CSV missing {missing}. Present: {list(df.columns)}")

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
    if hasattr(dino_model, "embed_global_batch"):
        out = dino_model.embed_global_batch(imgs)
    elif hasattr(dino_model, "embed_batch"):
        out = dino_model.embed_batch(imgs)
    else:
        tok = dino_model.embed_tokens_batch(imgs, pool=1)
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
        img = load_image_gray(img_path)
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


@dataclass
class QueryPool:
    qidx: int
    row_idx: int
    source: str
    vec_rows: np.ndarray
    patch_ids: np.ndarray
    ranks: np.ndarray
    geom_dist: np.ndarray


@dataclass
class PreparedPools:
    q_vecs: np.ndarray
    patch_vecs: np.ndarray
    pools: List[QueryPool]
    train_q: np.ndarray
    val_q: np.ndarray
    test_q: np.ndarray


def _build_soft_targets_from_distances(cfg: TrainingConfig, d: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    K = int(d.shape[0])
    if K == 0:
        return 1.0, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    finite = np.isfinite(d)
    if not np.any(finite):
        tau = 1.0
        logits = np.zeros((K,), dtype=np.float32)
        probs = np.full((K,), 1.0 / float(K), dtype=np.float32)
        return tau, logits, probs

    df = d[finite].astype(np.float64, copy=False)
    qlo = float(np.quantile(df, float(cfg.tau_q_lo)))
    qhi = float(np.quantile(df, float(cfg.tau_q_hi)))
    spread = max(qhi - qlo, 1e-6)
    tau = spread / float(cfg.tau_div)
    tau = float(np.clip(tau, float(cfg.tau_min), float(cfg.tau_max)))

    logits = np.full((K,), -np.inf, dtype=np.float64)
    logits[finite] = -d[finite] / tau
    mx = float(np.max(logits[finite]))
    ex = np.zeros((K,), dtype=np.float64)
    ex[finite] = np.exp(logits[finite] - mx)
    Z = float(ex.sum())
    probs = ex / Z if Z > 0 else np.full((K,), 1.0 / float(K), dtype=np.float64)
    return tau, logits.astype(np.float32), probs.astype(np.float32)


def _transform_distance_target(cfg: TrainingConfig, d: np.ndarray) -> np.ndarray:
    out = np.asarray(d, dtype=np.float32).copy()
    out = np.where(np.isfinite(out), out, float(cfg.distance_clip_max))
    if cfg.distance_target == "clipped":
        out = np.clip(out, 0.0, float(cfg.distance_clip_max))
    elif cfg.distance_target == "log1p":
        out = np.log1p(np.clip(out, 0.0, float(cfg.distance_clip_max)))
    return out.astype(np.float32, copy=False)


def build_query_pools(
    cfg: TrainingConfig,
    hits_df: pd.DataFrame,
    key_to_qidx: Dict[Tuple[int, str], int],
    patch_id_to_vecrow: Optional[Dict[int, int]],
) -> Tuple[List[QueryPool], np.ndarray]:
    n_queries = len(key_to_qidx)
    row_idx_of_q = np.full((n_queries,), -1, dtype=np.int64)

    df = hits_df.copy()
    df[cfg.col_rank] = df[cfg.col_rank].astype(np.int64)
    df = df[df[cfg.col_rank] <= int(cfg.train_topk)]

    grouped: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for i, r in df.iterrows():
        key = (int(r[cfg.col_row_idx]), str(r[cfg.col_source]))
        grouped[key].append(int(i))

    pools: List[QueryPool] = []
    dropped = 0

    for (row_idx, src), qidx in key_to_qidx.items():
        row_idx_of_q[qidx] = int(row_idx)
        idxs = grouped.get((int(row_idx), str(src)), [])
        if not idxs:
            dropped += 1
            continue

        cols = [cfg.col_patch_id, cfg.col_rank, cfg.col_geom_dist]
        d = df.loc[idxs, cols].copy()
        d[cfg.col_patch_id] = d[cfg.col_patch_id].astype(np.int64)
        d[cfg.col_rank] = d[cfg.col_rank].astype(np.int64)
        d[cfg.col_geom_dist] = pd.to_numeric(d[cfg.col_geom_dist], errors="coerce")
        d.sort_values(cfg.col_rank, ascending=True, inplace=True)
        d = d.drop_duplicates(subset=[cfg.col_patch_id], keep="first")

        pids = d[cfg.col_patch_id].to_numpy(dtype=np.int64)
        if patch_id_to_vecrow is None:
            vec_rows = pids.copy()
        else:
            vec_rows = np.asarray([patch_id_to_vecrow.get(int(pid), -1) for pid in pids], dtype=np.int64)

        geom = d[cfg.col_geom_dist].to_numpy(dtype=np.float32)
        ranks = d[cfg.col_rank].to_numpy(dtype=np.int64)
        ok = vec_rows >= 0
        if int(ok.sum()) < int(cfg.require_min_candidates):
            dropped += 1
            continue

        pools.append(
            QueryPool(
                qidx=int(qidx),
                row_idx=int(row_idx),
                source=str(src),
                vec_rows=vec_rows[ok],
                patch_ids=pids[ok],
                ranks=ranks[ok],
                geom_dist=geom[ok],
            )
        )

    _log(f"[Reranker] Built query pools: kept={len(pools)} dropped={dropped}")
    return pools, row_idx_of_q


class ListwisePoolDataset(Dataset):
    def __init__(
        self,
        cfg: TrainingConfig,
        q_vecs: np.ndarray,
        patch_vecs: np.ndarray,
        pools: List[QueryPool],
        q_indices: np.ndarray,
        *,
        mode: str,
    ):
        self.cfg = cfg
        self.q_vecs = q_vecs
        self.patch_vecs = patch_vecs
        self.pools = pools
        self.q_indices = q_indices.astype(np.int64)
        self.mode = str(mode)
        self.pool_by_qidx = {int(p.qidx): p for p in pools}

    def __len__(self) -> int:
        return int(self.q_indices.shape[0])

    def _rng_for(self, qidx: int) -> np.random.RandomState:
        if self.mode == "train":
            return np.random.RandomState()
        return np.random.RandomState(int(self.cfg.seed) + int(self.cfg.eval_seed_offset) + int(qidx))

    def _sample_indices_uniform(self, n_pool: int, target_k: int, rng: np.random.RandomState) -> np.ndarray:
        k = min(int(target_k), int(n_pool))
        if k == n_pool:
            idx = np.arange(n_pool, dtype=np.int64)
        else:
            idx = rng.choice(n_pool, size=k, replace=False).astype(np.int64)
        return idx

    def _sample_indices_stratified(self, pool: QueryPool, target_k: int, rng: np.random.RandomState) -> np.ndarray:
        n_pool = int(pool.vec_rows.shape[0])
        if n_pool <= target_k:
            return np.arange(n_pool, dtype=np.int64)

        ranks = pool.ranks
        top = np.where(ranks <= int(self.cfg.band_top_end))[0]
        mid = np.where((ranks > int(self.cfg.band_top_end)) & (ranks <= int(self.cfg.band_mid_end)))[0]
        tail = np.where(ranks > int(self.cfg.band_mid_end))[0]

        selected: List[int] = []
        remaining = target_k
        for band, want in [(top, int(self.cfg.sample_top_n)), (mid, int(self.cfg.sample_mid_n)), (tail, int(self.cfg.sample_tail_n))]:
            if remaining <= 0:
                break
            take = min(int(want), int(band.shape[0]), remaining)
            if take > 0:
                chosen = rng.choice(band, size=take, replace=False).astype(np.int64).tolist()
                selected.extend(chosen)
                remaining -= take

        if remaining > 0:
            universe = np.arange(n_pool, dtype=np.int64)
            used = np.asarray(sorted(set(selected)), dtype=np.int64) if selected else np.zeros((0,), dtype=np.int64)
            mask = np.ones((n_pool,), dtype=bool)
            if used.size > 0:
                mask[used] = False
            leftover = universe[mask]
            if leftover.size > 0:
                take = min(remaining, int(leftover.shape[0]))
                chosen = rng.choice(leftover, size=take, replace=False).astype(np.int64).tolist()
                selected.extend(chosen)

        idx = np.asarray(selected, dtype=np.int64)
        if idx.size == 0:
            return self._sample_indices_uniform(n_pool, target_k, rng)
        if idx.size > target_k:
            idx = idx[:target_k]
        return idx

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        qidx = int(self.q_indices[i])
        pool = self.pool_by_qidx[qidx]
        q = np.asarray(self.q_vecs[qidx], dtype=np.float32)
        n_pool = int(pool.vec_rows.shape[0])

        if self.mode != "train" and bool(self.cfg.eval_use_full_list):
            sel = np.arange(n_pool, dtype=np.int64)
        else:
            target_k = min(int(self.cfg.list_k), n_pool)
            rng = self._rng_for(qidx)
            if self.mode == "train" and self.cfg.sampling_mode == "stratified":
                sel = self._sample_indices_stratified(pool, target_k, rng)
            else:
                sel = self._sample_indices_uniform(n_pool, target_k, rng)

        if self.mode == "train" and bool(self.cfg.shuffle_candidates):
            rng = self._rng_for(qidx)
            rng.shuffle(sel)

        rows = pool.vec_rows[sel]
        c = np.asarray(self.patch_vecs[rows], dtype=np.float32)
        d = np.asarray(pool.geom_dist[sel], dtype=np.float32)
        _, _, p = _build_soft_targets_from_distances(self.cfg, d)
        dt = _transform_distance_target(self.cfg, d)
        k = int(sel.shape[0])
        mask = np.ones((k,), dtype=np.bool_)

        return {
            "q_emb": torch.from_numpy(q),
            "c_emb": torch.from_numpy(c),
            "gt_prob": torch.from_numpy(np.asarray(p, dtype=np.float32)),
            "gt_dist": torch.from_numpy(np.asarray(dt, dtype=np.float32)),
            "mask": torch.from_numpy(mask),
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


def prepare_pools(cfg: TrainingConfig, *, embed_dim: int) -> PreparedPools:
    cfg.validate()
    hits_df = load_hits_csv(cfg)
    dataset_df = load_dataset_csv(cfg)
    patch_vecs = load_patch_vectors(cfg.patch_vectors_path)
    patch_id_to_vecrow = load_patch_id_mapping(cfg.patch_manifest_path, n_vectors=int(patch_vecs.shape[0]))
    q_vecs, key_to_qidx = build_or_load_query_vectors(cfg, hits_df, dataset_df, embed_dim=embed_dim)
    pools, row_idx_of_q = build_query_pools(cfg, hits_df, key_to_qidx, patch_id_to_vecrow)

    kept_qidx = np.asarray([int(p.qidx) for p in pools], dtype=np.int64)
    kept_rows = np.asarray([int(p.row_idx) for p in pools], dtype=np.int64)
    train_rows, val_rows, test_rows = _split_by_row_idx(cfg, kept_rows)

    train_q = kept_qidx[np.isin(kept_rows, np.fromiter(train_rows, dtype=np.int64))]
    val_q = kept_qidx[np.isin(kept_rows, np.fromiter(val_rows, dtype=np.int64))]
    test_q = kept_qidx[np.isin(kept_rows, np.fromiter(test_rows, dtype=np.int64))]

    _log(
        "[Reranker] Dataset sizes:\n"
        f"  - train queries: {len(train_q)}\n"
        f"  - val queries  : {len(val_q)}\n"
        f"  - test queries : {len(test_q)}"
    )

    return PreparedPools(
        q_vecs=q_vecs,
        patch_vecs=patch_vecs,
        pools=pools,
        train_q=train_q,
        val_q=val_q,
        test_q=test_q,
    )


def prepare_dataloaders(cfg: TrainingConfig, *, embed_dim: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    prep = prepare_pools(cfg, embed_dim=embed_dim)
    train_ds = ListwisePoolDataset(cfg, prep.q_vecs, prep.patch_vecs, prep.pools, prep.train_q, mode="train")
    val_ds = ListwisePoolDataset(cfg, prep.q_vecs, prep.patch_vecs, prep.pools, prep.val_q, mode="val")
    test_ds = ListwisePoolDataset(cfg, prep.q_vecs, prep.patch_vecs, prep.pools, prep.test_q, mode="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
