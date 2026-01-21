# index/reranker/model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RerankerConfig


class PairMLP(nn.Module):
    """
    Scores (q_emb, c_emb) pairs.

    Inputs:
      q_emb: (B, D)
      c_emb: (B, K, D)

    Output:
      scores: (B, K)  (higher = better)
    """

    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.embed_dim)

        # Feature construction
        base_dim = 4 * D if bool(cfg.use_pair_features) else 2 * D
        scalar_dim = 3 if bool(cfg.add_scalar_features) else 0  # cosine, ||q||, ||c||
        in_dim = base_dim + scalar_dim

        h1, h2 = cfg.hidden_dims

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, int(h1)))
        if bool(cfg.use_layernorm):
            layers.append(nn.LayerNorm(int(h1)))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(float(cfg.dropout)))

        layers.append(nn.Linear(int(h1), int(h2)))
        if bool(cfg.use_layernorm):
            layers.append(nn.LayerNorm(int(h2)))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(float(cfg.dropout)))

        layers.append(nn.Linear(int(h2), 1))
        self.net = nn.Sequential(*layers)

        # Optional query-dependent logit scaling
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

    def forward(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        if q_emb.ndim != 2:
            raise ValueError(f"q_emb must be (B,D), got {q_emb.shape}")
        if c_emb.ndim != 3:
            raise ValueError(f"c_emb must be (B,K,D), got {c_emb.shape}")

        B, K, D = c_emb.shape

        # Expand query to list dimension
        q = q_emb[:, None, :].expand(B, K, D)  # (B,K,D)

        # Optional embedding normalization before feature construction
        q, c = self._maybe_norm(q, c_emb)

        # Base pair features
        if bool(self.cfg.use_pair_features):
            x = torch.cat([q, c, (q - c).abs(), q * c], dim=-1)  # (B,K,4D)
        else:
            x = torch.cat([q, c], dim=-1)  # (B,K,2D)

        # Optional scalar features (helps anchor “visual similarity” from cosine)
        if bool(self.cfg.add_scalar_features):
            # cosine similarity in [-1,1] (if normalized -> exact cosine)
            cos = (q * c).sum(dim=-1, keepdim=True)  # (B,K,1)

            # norms (if normalized -> ~1, but still ok)
            qn = torch.linalg.vector_norm(q, ord=2, dim=-1, keepdim=True)  # (B,K,1)
            cn = torch.linalg.vector_norm(c, ord=2, dim=-1, keepdim=True)  # (B,K,1)

            x = torch.cat([x, cos, qn, cn], dim=-1)  # (B,K,in_dim)

        scores = self.net(x).squeeze(-1)  # (B,K)

        # Optional query-dependent scale (stabilizes listwise softmax)
        if self.use_query_scale:
            # scale per query (B,1) -> expand to (B,K)
            s = self.q_scale(q_emb).clamp(-5.0, 5.0).exp()  # positive
            scores = scores * s

        return scores


class ListwiseReranker(nn.Module):
    def __init__(self, cfg: Optional[RerankerConfig] = None):
        super().__init__()
        self.cfg = cfg or RerankerConfig()
        self.head = PairMLP(self.cfg)
        self.to(self.cfg.device)

    def forward(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        return self.head(q_emb, c_emb)

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
        model = cls(cfg=cfg)
        model.load_state_dict(obj["state_dict"])
        model.to(cfg.device)
        print(f"[Reranker] Loaded from {path}")
        return model
