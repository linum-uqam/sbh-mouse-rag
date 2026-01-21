# index/reranker/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple


@dataclass
class TrainingConfig:
    # Inputs
    hits_csv: str = "out/reranker_dataset/eval_hits.csv"
    dataset_csv: str = "out/reranker_dataset/dataset.csv"

    # Candidate embedding store
    patch_manifest_path: str = "out/index/patch_manifest.parquet"  # optional
    patch_vectors_path: str = "out/index/patch_vectors.npy"

    # Query embedding cache (computed once from dataset images)
    query_vectors_cache: str = "out/reranker/query_vectors.npy"

    # Output
    out_path: str = "out/reranker/reranker_listwise.pt"

    # Candidate list settings
    train_topk: int = 100           # consider ranks
    list_k: int = 100               # fixed list size per query (pad if fewer)
    require_min_candidates: int = 2  # drop queries with fewer valid candidates after mapping

    # Column names expected in eval_hits.csv
    col_row_idx: str = "row_idx"
    col_source: str = "source"
    col_patch_id: str = "patch_id"
    col_rank: str = "rank"
    col_gt_prob: str = "gt_prob"    # preferred
    col_gt_logit: str = "gt_logit"  # fallback (optional)

    # Split (by row_idx to avoid leakage between sources)
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42

    # Training
    batch_size: int = 64
    num_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    num_workers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RerankerConfig:
    # Model backbone inputs
    embed_dim: int = 768

    # MLP head
    hidden_dims: Tuple[int, int] = (512, 256)
    dropout: float = 0.15

    # Device placement (used by model.load/save convenience)
    device: str = "cuda"

    # Pair feature type: concat(q, c, |q-c|, q*c) = 4D features
    use_pair_features: bool = True

    # -----------------------------
    # Moved from "recommended defaults" in model.py
    # -----------------------------
    normalize_embeddings: bool = True
    add_scalar_features: bool = True   # e.g. cosine(q,c), ||q||, ||c||
    use_layernorm: bool = True

    # Optional extras
    use_query_scale: bool = True
    query_scale_hidden: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RerankerConfig":
        return RerankerConfig(**d)
