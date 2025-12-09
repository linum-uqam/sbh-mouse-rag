# index/reranker/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch
import torch.nn.functional as F

from index.utils import log
from index.model.dino import model as dino_model  # shared encoder
from .model import TwoTowerReranker


@dataclass
class RerankerRuntimeConfig:
    """
    Runtime config for using the trained reranker.

    Attributes
    ----------
    model_path:
        Path to the trained reranker weights (.pt) produced by train_reranker.py.
    device:
        Device to run inference on ("cuda" or "cpu").
    batch_size:
        Unused for embedding-only scoring (kept for API compat).
    """
    model_path: Path = Path("out/reranker/reranker.pt")
    device: str = "cuda"
    batch_size: int = 32


class RerankerService:
    """
    Thin runtime wrapper around TwoTowerReranker for inference.

    Embedding-only API (no images, no AllenSDK):

      - score_emb_pairs(q_emb, cand_embs) -> np.ndarray

        q_emb can be:
          - shape (D,)  -> broadcast to all candidates
          - shape (N,D) -> one query embedding per candidate
        cand_embs must be shape (N,D).
    """

    def __init__(self, cfg: RerankerRuntimeConfig):
        self.cfg = cfg
        self.model = TwoTowerReranker.load(
            cfg.model_path,
            encoder=dino_model,
            map_location=cfg.device,
        )
        self.model.eval()
        log(
            f"RerankerService | loaded model from {cfg.model_path} "
            f"on device={cfg.device}"
        )

    # ---- embedding-based API ----

    def score_emb_pairs(
        self,
        query_emb: np.ndarray,
        cand_embs: np.ndarray,
    ) -> np.ndarray:
        """
        Score (query, candidate) pairs using embeddings only.

        Parameters
        ----------
        query_emb:
            - np.ndarray (D,)      -> same query for all candidates
            - np.ndarray (N,D)     -> per-candidate query embedding
        cand_embs:
            - np.ndarray (N,D)     -> candidate embeddings

        Returns
        -------
        scores:
            np.ndarray (N,) with higher = more relevant.
        """
        C = np.asarray(cand_embs, dtype=np.float32)
        if C.ndim != 2:
            raise ValueError(f"cand_embs must be (N,D), got shape {C.shape}")
        N, D = C.shape

        Q = np.asarray(query_emb, dtype=np.float32)
        if Q.ndim == 1:
            if Q.shape[0] != D:
                raise ValueError(f"query_emb dim {Q.shape[0]} != cand_embs dim {D}")
            # broadcast single query to all candidates
            Q = np.broadcast_to(Q, (N, D)).copy()
        elif Q.ndim == 2:
            if Q.shape != C.shape:
                raise ValueError(
                    f"query_emb shape {Q.shape} must match cand_embs shape {C.shape} "
                    "for per-candidate scoring"
                )
        else:
            raise ValueError(f"query_emb must be (D,) or (N,D), got shape {Q.shape}")

        device = self.model.device_name  # set in TwoTowerReranker.__init__

        Q_t = torch.from_numpy(Q).to(device)
        C_t = torch.from_numpy(C).to(device)

        # Ensure L2-normalization (safe even if already normalized)
        Q_t = F.normalize(Q_t, dim=-1)
        C_t = F.normalize(C_t, dim=-1)

        self.model.eval()
        with torch.no_grad():
            scores_t = self.model(Q_t, C_t)  # forward(q_emb, c_emb) -> (N,)

        scores = scores_t.detach().cpu().numpy().astype(np.float32)
        return scores
