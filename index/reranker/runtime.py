from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from index.utils import log
from .model import ListwiseReranker


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    """
    Runtime config for using the trained reranker.
    """
    model_path: Path = Path("out/reranker/reranker_listwise.pt")
    device: str = "cuda"
    batch_size: int = 256

    # Optional runtime behaviors
    normalize_embeddings: bool = True      # L2-normalize q/c before scoring (usually good with cosine-trained encoders)
    use_fp16: bool = True                 # autocast on CUDA
    compile_model: bool = False           # torch.compile if available (PyTorch 2+)


class RerankerService:
    """
    Runtime wrapper around ListwiseReranker.

    Primary use:
      - score_list(query_emb (D,), cand_embs (N,D)) -> scores (N,)

    Back-compat:
      - score_emb_pairs(query_emb, cand_embs) -> np.ndarray (N,)
        accepts:
          query_emb: (D,), (1,D), or (N,D)
          cand_embs: (N,D)
    """

    def __init__(self, cfg: RerankerRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Load model (saved with cfg inside)
        self.model = ListwiseReranker.load(cfg.model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        if cfg.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                log("[Reranker] torch.compile enabled")
            except Exception as e:
                log(f"[Reranker] torch.compile failed (ignored): {e}")

        # Use config from model if present (fallback to runtime cfg)
        self._model_cfg = getattr(self.model, "cfg", None)

        log(f"RerankerService | loaded model={cfg.model_path} device={cfg.device}")

    # -------------------------
    # Public API
    # -------------------------

    def score_emb_pairs(self, query_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
        """
        Back-compat entrypoint.
        """
        cand_embs = self._as_2d_f32(cand_embs, name="cand_embs")
        q = self._as_f32(query_emb)

        N, D = cand_embs.shape

        # Normalize input shapes
        if q.ndim == 1:
            if q.shape[0] != D:
                raise ValueError(f"query_emb (D,) must match D={D}, got {q.shape}")
            return self.score_list(q, cand_embs)

        if q.ndim == 2:
            if q.shape[1] != D:
                raise ValueError(f"query_emb (N,D) must match D={D}, got {q.shape}")

            if q.shape[0] == 1:
                return self.score_list(q.reshape(-1), cand_embs)

            if q.shape[0] == N:
                return self.score_pairs(q, cand_embs)

            raise ValueError(f"query_emb first dim must be 1 or N={N}, got {q.shape}")

        raise ValueError(f"query_emb must be (D,) or (N,D), got {q.shape}")

    def score_list(self, query_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
        """
        Score a single query against a candidate list.

        query_emb: (D,)
        cand_embs: (N,D)
        returns:   (N,) float32, higher=better
        """
        cand_embs = self._as_2d_f32(cand_embs, name="cand_embs")
        q = self._as_1d_f32(query_emb, name="query_emb")

        N, D = cand_embs.shape
        if q.shape[0] != D:
            raise ValueError(f"query_emb dim must match D={D}, got {q.shape}")

        # Torch tensors (keep on device)
        q_t = torch.from_numpy(q).to(self.device)                 # (D,)
        c_t = torch.from_numpy(cand_embs).to(self.device)         # (N,D)

        if self.cfg.normalize_embeddings:
            q_t = torch.nn.functional.normalize(q_t, dim=0)
            c_t = torch.nn.functional.normalize(c_t, dim=1)

        bs = int(max(1, self.cfg.batch_size))
        out = np.empty((N,), dtype=np.float32)

        use_autocast = (self.cfg.use_fp16 and self.device.type == "cuda")

        with torch.no_grad():
            for s in range(0, N, bs):
                e = min(s + bs, N)
                c_chunk = c_t[s:e]                                # (K,D)
                # Listwise model expects (B,D) and (B,K,D)
                q_batch = q_t.unsqueeze(0)                         # (1,D)
                c_batch = c_chunk.unsqueeze(0)                     # (1,K,D)

                with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
                    scores = self.model(q_batch, c_batch).squeeze(0)  # (K,)

                out[s:e] = scores.float().cpu().numpy()

        return out

    def score_pairs(self, query_embs: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
        """
        Score aligned pairs (q_i, c_i). This is mostly for debugging / niche use.

        query_embs: (N,D)
        cand_embs:  (N,D)
        returns:    (N,)
        """
        q = self._as_2d_f32(query_embs, name="query_embs")
        c = self._as_2d_f32(cand_embs, name="cand_embs")

        if q.shape != c.shape:
            raise ValueError(f"query_embs and cand_embs must match, got {q.shape} vs {c.shape}")

        N, D = c.shape
        q_t = torch.from_numpy(q).to(self.device)                # (N,D)
        c_t = torch.from_numpy(c).to(self.device)                # (N,D)

        if self.cfg.normalize_embeddings:
            q_t = torch.nn.functional.normalize(q_t, dim=1)
            c_t = torch.nn.functional.normalize(c_t, dim=1)

        bs = int(max(1, self.cfg.batch_size))
        out = np.empty((N,), dtype=np.float32)
        use_autocast = (self.cfg.use_fp16 and self.device.type == "cuda")

        with torch.no_grad():
            for s in range(0, N, bs):
                e = min(s + bs, N)
                q_chunk = q_t[s:e]                               # (B,D)
                c_chunk = c_t[s:e].unsqueeze(1)                  # (B,1,D)

                with torch.cuda.amp.autocast(enabled=use_autocast):
                    scores = self.model(q_chunk, c_chunk).squeeze(1)  # (B,)

                out[s:e] = scores.float().cpu().numpy()

        return out

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _as_f32(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return x

    def _as_1d_f32(self, x: np.ndarray, *, name: str) -> np.ndarray:
        x = self._as_f32(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.reshape(-1)
        if x.ndim != 1:
            raise ValueError(f"{name} must be (D,), got {x.shape}")
        return x

    def _as_2d_f32(self, x: np.ndarray, *, name: str) -> np.ndarray:
        x = self._as_f32(x)
        if x.ndim != 2:
            raise ValueError(f"{name} must be (N,D), got {x.shape}")
        return x
