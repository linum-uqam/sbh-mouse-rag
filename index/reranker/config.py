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

    # Pool / list settings
    train_topk: int = 100
    list_k: int = 64
    require_min_candidates: int = 8
    eval_use_full_list: bool = False
    eval_seed_offset: int = 10_000

    # Sampling
    sampling_mode: str = "stratified"  # stratified | uniform
    shuffle_candidates: bool = True
    sample_top_n: int = 16
    sample_mid_n: int = 24
    sample_tail_n: int = 24
    band_top_end: int = 10
    band_mid_end: int = 40

    # Soft-target recompute from geom_dist_vox
    tau_q_lo: float = 0.10
    tau_q_hi: float = 0.90
    tau_div: float = 4.0
    tau_min: float = 1.0
    tau_max: float = 256.0

    # Distance supervision
    use_distance_loss: bool = True
    distance_loss_weight: float = 0.25
    distance_loss_type: str = "huber"  # huber | mse
    distance_target: str = "log1p"  # raw | clipped | log1p
    distance_clip_max: float = 256.0

    # Column names expected in eval_hits.csv
    col_row_idx: str = "row_idx"
    col_source: str = "source"
    col_patch_id: str = "patch_id"
    col_rank: str = "rank"
    col_geom_dist: str = "geom_dist_vox"
    col_gt_prob: str = "gt_prob"
    col_gt_logit: str = "gt_logit"

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

    # Early stopping / scheduler
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.001
    use_plateau_scheduler: bool = True
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6

    def validate(self) -> None:
        if int(self.list_k) <= 0:
            raise ValueError("list_k must be > 0")
        if int(self.train_topk) <= 0:
            raise ValueError("train_topk must be > 0")
        if self.sampling_mode not in {"stratified", "uniform"}:
            raise ValueError(f"Unsupported sampling_mode={self.sampling_mode!r}")
        if self.distance_loss_type not in {"huber", "mse"}:
            raise ValueError(f"Unsupported distance_loss_type={self.distance_loss_type!r}")
        if self.distance_target not in {"raw", "clipped", "log1p"}:
            raise ValueError(f"Unsupported distance_target={self.distance_target!r}")
        if int(self.early_stopping_patience) < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if float(self.early_stopping_min_delta) < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if int(self.plateau_patience) < 0:
            raise ValueError("plateau_patience must be >= 0")
        if not (0.0 < float(self.plateau_factor) < 1.0):
            raise ValueError("plateau_factor must be in (0,1)")
        if float(self.plateau_min_lr) < 0:
            raise ValueError("plateau_min_lr must be >= 0")
        s = int(self.sample_top_n) + int(self.sample_mid_n) + int(self.sample_tail_n)
        if self.sampling_mode == "stratified" and s != int(self.list_k):
            raise ValueError(
                "For stratified sampling, sample_top_n + sample_mid_n + sample_tail_n must equal list_k. "
                f"Got {self.sample_top_n}+{self.sample_mid_n}+{self.sample_tail_n}!={self.list_k}"
            )

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

    normalize_embeddings: bool = True
    add_scalar_features: bool = True
    use_layernorm: bool = True

    use_query_scale: bool = True
    query_scale_hidden: int = 128

    # Dual-head model
    use_distance_head: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RerankerConfig":
        return RerankerConfig(**d)
