from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RerankerConfig


class PairMLP(nn.Module):
    """
    Scores (q_emb, c_emb) pairs and optionally predicts a distance target.

    Inputs:
      q_emb: (B, D)
      c_emb: (B, K, D)

    Output:
      rank_scores: (B, K)  (higher = better)
      dist_pred:   (B, K)  (larger = farther)
    """

    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.embed_dim)

        base_dim = 4 * D if bool(cfg.use_pair_features) else 2 * D
        scalar_dim = 3 if bool(cfg.add_scalar_features) else 0
        in_dim = base_dim + scalar_dim
        h1, h2 = cfg.hidden_dims

        trunk: list[nn.Module] = [nn.Linear(in_dim, int(h1))]
        if bool(cfg.use_layernorm):
            trunk.append(nn.LayerNorm(int(h1)))
        trunk += [nn.GELU(), nn.Dropout(float(cfg.dropout)), nn.Linear(int(h1), int(h2))]
        if bool(cfg.use_layernorm):
            trunk.append(nn.LayerNorm(int(h2)))
        trunk += [nn.GELU(), nn.Dropout(float(cfg.dropout))]
        self.trunk = nn.Sequential(*trunk)

        self.rank_head = nn.Linear(int(h2), 1)
        self.use_distance_head = bool(getattr(cfg, "use_distance_head", True))
        self.dist_head = nn.Linear(int(h2), 1) if self.use_distance_head else None

        self.use_query_scale = bool(getattr(cfg, "use_query_scale", False))
        if self.use_query_scale:
            qs_h = int(getattr(cfg, "query_scale_hidden", 128))
            self.q_scale = nn.Sequential(
                nn.Linear(D, qs_h),
                nn.GELU(),
                nn.Linear(qs_h, 1),
            )

    def _maybe_norm(self, q: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.cfg.normalize_embeddings):
            q = F.normalize(q, p=2, dim=-1)
            c = F.normalize(c, p=2, dim=-1)
        return q, c

    def forward(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q_emb.ndim != 2:
            raise ValueError(f"q_emb must be (B,D), got {q_emb.shape}")
        if c_emb.ndim != 3:
            raise ValueError(f"c_emb must be (B,K,D), got {c_emb.shape}")

        B, K, D = c_emb.shape
        q = q_emb[:, None, :].expand(B, K, D)
        q, c = self._maybe_norm(q, c_emb)

        if bool(self.cfg.use_pair_features):
            x = torch.cat([q, c, (q - c).abs(), q * c], dim=-1)
        else:
            x = torch.cat([q, c], dim=-1)

        if bool(self.cfg.add_scalar_features):
            cos = (q * c).sum(dim=-1, keepdim=True)
            qn = torch.linalg.vector_norm(q, ord=2, dim=-1, keepdim=True)
            cn = torch.linalg.vector_norm(c, ord=2, dim=-1, keepdim=True)
            x = torch.cat([x, cos, qn, cn], dim=-1)

        h = self.trunk(x)
        rank_scores = self.rank_head(h).squeeze(-1)

        if self.use_query_scale:
            s = self.q_scale(q_emb).clamp(-5.0, 5.0).exp()
            rank_scores = rank_scores * s

        if self.dist_head is None:
            dist_pred = -rank_scores
        else:
            dist_pred = self.dist_head(h).squeeze(-1)
        return rank_scores, dist_pred


class ListwiseReranker(nn.Module):
    def __init__(self, cfg: Optional[RerankerConfig] = None):
        super().__init__()
        self.cfg = cfg or RerankerConfig()
        self.head = PairMLP(self.cfg)
        self.to(self.cfg.device)

    def forward(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(q_emb, c_emb)

    def rank(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        rank_scores, _ = self.forward(q_emb, c_emb)
        return rank_scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        obj = {"cfg": self.cfg.to_dict(), "state_dict": self.state_dict()}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        print(f"[Reranker] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ListwiseReranker":
        path = Path(path)
        obj = torch.load(path, map_location=map_location)
        cfg = RerankerConfig.from_dict(obj["cfg"])
        runtime_device = str(map_location)
        cfg.device = runtime_device
        model = cls(cfg=cfg)
        model.load_state_dict(obj["state_dict"])
        model.to(runtime_device)
        model.eval()
        print(f"[Reranker] Loaded from {path} on device={runtime_device}")
        return model
