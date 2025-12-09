# index/reranker/model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol, Sequence, Iterable, Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from index.model.dino import model as dino_model  # your existing encoder
from index.utils import log  # or replace with print


class ImageEncoder(Protocol):
    """
    Minimal protocol for a vision encoder used by the reranker.

    It only requires:
      - embed(img: np.ndarray) -> np.ndarray  (1D vector)
    """

    def embed(self, img: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class RerankerConfig:
    """
    Configuration for the two-tower reranker head.

    Attributes
    ----------
    embed_dim:
        Dimension of the encoder embeddings (e.g., 768 for ViT-B/16).
    hidden_dims:
        Hidden layer sizes for the MLP head.
    device:
        Default device ("cuda", "cpu", etc.).
    """
    embed_dim: int = 768
    hidden_dims: Sequence[int] = (512, 256)
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RerankerConfig":
        return RerankerConfig(**d)


class MLPHead(nn.Module):
    """
    Simple MLP over concatenated features derived from (q_emb, c_emb).

    Input feature = concat( q, c, |q-c|, q*c )  -> scalar score.
    """

    def __init__(self, embed_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        in_dim = 4 * embed_dim
        layers: List[nn.Module] = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # q, c: (B, D)
        diff = torch.abs(q - c)
        prod = q * c
        x = torch.cat([q, c, diff, prod], dim=-1)  # (B, 4D)
        out = self.net(x).squeeze(-1)  # (B,)
        return out


class TwoTowerReranker(nn.Module):
    """
    Two-tower image reranker.

    - Uses a shared encoder to convert query and candidate images into embeddings.
    - Uses an MLP head over (q_emb, c_emb) to predict a scalar relevance score.

    The encoder is **frozen** by default (we treat it as a feature extractor only),
    so gradients only flow through the MLP head.
    """

    def __init__(
        self,
        cfg: Optional[RerankerConfig] = None,
        encoder: Optional[ImageEncoder] = None,
    ):
        super().__init__()
        self.cfg = cfg or RerankerConfig()
        self.device_name = self.cfg.device

        # encoder is a simple Python object, not a nn.Module (we treat it as frozen).
        # Default = your existing DINO encoder.
        self.encoder: ImageEncoder = encoder or dino_model

        self.head = MLPHead(
            embed_dim=self.cfg.embed_dim,
            hidden_dims=self.cfg.hidden_dims,
        )
        self.to(self.device_name)

    # ----------------- core API -----------------

    @torch.no_grad()
    def score_emb_pairs(
        self,
        q_emb: np.ndarray,
        cand_embs: np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Score (query, candidate) pairs given embeddings only.

        Parameters
        ----------
        q_emb:
            Query embedding, shape (D,) or (1,D).
        cand_embs:
            Candidate embeddings, shape (N,D).
        batch_size:
            Batch size for running the head.

        Returns
        -------
        scores: np.ndarray of shape (N,), higher = more relevant.
        """
        self.eval()

        q = np.asarray(q_emb, dtype=np.float32)
        C = np.asarray(cand_embs, dtype=np.float32)

        if q.ndim == 1:
            q = q[None, :]  # (1,D)

        if C.ndim != 2:
            raise ValueError(f"cand_embs must be (N,D), got shape {C.shape}")

        N, D = C.shape
        if q.shape[1] != D:
            raise ValueError(
                f"Dim mismatch: q_emb has D={q.shape[1]}, cand_embs has D={D}"
            )

        # to torch, on correct device, L2-normalize
        q_t = torch.from_numpy(q).to(self.device_name)
        C_t = torch.from_numpy(C).to(self.device_name)

        q_t = F.normalize(q_t, dim=-1)
        C_t = F.normalize(C_t, dim=-1)

        scores_all: List[np.ndarray] = []

        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            c_batch = C_t[start:end]  # (b,D)

            # repeat query for this batch
            q_batch = q_t.expand(c_batch.size(0), -1)  # (b,D)

            s = self.forward(q_batch, c_batch)  # (b,)
            scores_all.append(s.cpu().numpy())

        if not scores_all:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(scores_all, axis=0)
    
    @torch.no_grad()
    def encode_batch(
        self,
        images: Union[Sequence[np.ndarray], torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode a batch of images with the shared encoder.

        Parameters
        ----------
        images:
            Either:
              - Sequence of np.ndarray (H,W) or (H,W,3) in [0,1] or [0,255], or
              - torch.Tensor of shape (B,H,W[,...]) from a DataLoader.

        Returns
        -------
        torch.Tensor of shape (B, D) on self.device_name.
        """
        # Normalize to list[np.ndarray]
        if isinstance(images, torch.Tensor):
            arr = images.detach().cpu().numpy()
            imgs: List[np.ndarray] = [arr[i] for i in range(arr.shape[0])]
        else:
            imgs = list(images)

        vecs: List[np.ndarray] = []
        for img in imgs:
            # encoder.embed returns np.ndarray (D,)
            v = self.encoder.embed(img)
            vecs.append(v.astype(np.float32, copy=False))

        arr = np.stack(vecs, axis=0)  # (B, D)
        emb = torch.from_numpy(arr).to(self.device_name)
        # Optionally L2-normalize here if encoder doesn't already.
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass given embeddings.

        Parameters
        ----------
        q_emb: (B, D)
        c_emb: (B, D)

        Returns
        -------
        scores: (B,) tensor of predicted relevance.
        """
        return self.head(q_emb, c_emb)

    def score_pairs(
        self,
        query_images: Sequence[np.ndarray],
        candidate_images: Sequence[np.ndarray],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Convenience method for inference: image pairs -> scores.

        Parameters
        ----------
        query_images:
            Sequence of query images (np.ndarray).
        candidate_images:
            Sequence of candidate images (same length as query_images).
        batch_size:
            Batch size for encoding.

        Returns
        -------
        scores: np.ndarray of shape (N,)
        """
        assert len(query_images) == len(candidate_images), "q/c length mismatch"
        N = len(query_images)
        scores_all: List[np.ndarray] = []

        self.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(N, start + batch_size)
                q_batch = query_images[start:end]
                c_batch = candidate_images[start:end]

                q_emb = self.encode_batch(q_batch)  # (B,D)
                c_emb = self.encode_batch(c_batch)  # (B,D)

                s = self.forward(q_emb, c_emb)  # (B,)
                scores_all.append(s.cpu().numpy())

        return np.concatenate(scores_all, axis=0)

    # ----------------- persistence -----------------

    def save(self, path: str | Path) -> None:
        """
        Save the reranker head weights + config.

        The encoder is not saved here (we assume it's external / reusable).
        """
        path = Path(path)
        obj = {
            "cfg": self.cfg.to_dict(),
            "state_dict": self.state_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        log(f"[Reranker] Saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: Optional[ImageEncoder] = None,
        map_location: str | torch.device = "cpu",
    ) -> "TwoTowerReranker":
        """
        Load reranker head + config from disk.

        You must pass an encoder compatible with the original training setup.
        """
        path = Path(path)
        obj = torch.load(path, map_location=map_location)
        cfg = RerankerConfig.from_dict(obj["cfg"])
        model = cls(cfg=cfg, encoder=encoder)
        model.load_state_dict(obj["state_dict"])
        model.to(cfg.device)
        log(f"[Reranker] Loaded from {path}")
        return model
