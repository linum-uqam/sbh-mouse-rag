# index/model/dino.py
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class _HFVisionEncoder:
    """
    Minimal DINOv3 wrapper (ViT-Base/16 by default) with:
      - embed(img) -> (D,)
      - embed_tokens(img) -> (T,D)  [patch tokens only; no CLS; T = H*W]
      - embed_both(img) -> {"global": (D,), "tokens": (T,D)}
      - embed_query_rotations(img, angles=[0,90,180,270]) -> List[{"angle","global","tokens"}]
      - embed_query_augmentations(img, angles, scales) -> rotations, each with pooled (k×k) query tokens + masks

    Notes
    -----
    - Inputs can be grayscale (H,W) or RGB (H,W,3), in [0,1] or [0,255].
    - Globals and tokens are L2-normalized row-wise (tokens).
    - Token grid is square with side `grid_hw`.
    """

    # ---------------- Initialization ----------------

    def __init__(self, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # HF components
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id, dtype=self.dtype)
        self.encoder.to(self.device, dtype=self.dtype).eval()

        # geometry
        self.image_size = int(getattr(self.encoder.config, "image_size", 224))
        self.patch_size = int(getattr(self.encoder.config, "patch_size", 16))
        self.grid_hw = self.image_size // self.patch_size  # expected H == W
        print(
            f"Using {model_id} on {self.device} | "
            f"img={self.image_size}, patch={self.patch_size}, grid={self.grid_hw}x{self.grid_hw}"
        )

    # ---------------- Image I/O ----------------

    @staticmethod
    def _to_pil_rgb(img_np: np.ndarray) -> Image.Image:
        """(H,W) or (H,W,1/3) in [0,1] or uint8 -> PIL RGB."""
        if img_np.ndim == 2:  # grayscale
            x = img_np
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            return Image.fromarray(x.astype(np.uint8), mode="L").convert("RGB")

        if img_np.ndim == 3 and img_np.shape[2] in (1, 3):
            x = img_np
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            x = x.astype(np.uint8)
            if x.shape[2] == 1:
                return Image.fromarray(x.squeeze(-1), mode="L").convert("RGB")
            return Image.fromarray(x, mode="RGB")

        raise ValueError("Expected (H,W) or (H,W,1/3) numpy array")

    def _prep_batch(self, pil_list: List[Image.Image]) -> torch.Tensor:
        """Processor handles resize/crop/normalize to model expected size."""
        return self.processor(images=pil_list, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)

    # ---------------- Token / Forward utils ----------------

    def _patch_tokens_only(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, L, D)
        return: (B, H*W, D) with ONLY patch tokens (drop CLS/extras)
        """
        B, L, D = last_hidden.shape
        need = self.grid_hw * self.grid_hw

        if L >= need + 1:
            patches = last_hidden[:, 1:1 + need, :]
        elif L == need:
            patches = last_hidden
        elif L > need:
            patches = last_hidden[:, L - need:L, :]
        else:
            raise ValueError(f"Not enough tokens: L={L}, need≥{need} (grid={self.grid_hw}x{self.grid_hw})")

        assert patches.shape[1] == need, f"Token trim mismatch: {patches.shape[1]} vs expected {need}"
        return patches

    def _global_from_hidden(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Prefer CLS if present; else mean of patch tokens."""
        B, L, D = last_hidden.shape
        need = self.grid_hw * self.grid_hw
        if L >= need + 1:
            return last_hidden[:, 0, :]  # CLS
        patches = self._patch_tokens_only(last_hidden)
        return patches.mean(dim=1)

    @torch.inference_mode()
    def _forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict of tensors on CPU float32:
          - global: (B,D)
          - tokens: (B, H*W, D)
        """
        out = self.encoder(pixel_values=pixel_values)
        hidden = out.last_hidden_state  # (B, L, D)

        g = self._global_from_hidden(hidden)           # (B,D)
        p = self._patch_tokens_only(hidden)            # (B, H*W, D)

        # L2 normalize on device → cast to float32 on CPU
        g = F.normalize(g, p=2, dim=-1).to(dtype=torch.float32).cpu()
        p = F.normalize(p, p=2, dim=-1).to(dtype=torch.float32).cpu()
        return {"global": g, "tokens": p}

    @staticmethod
    def _l2norm_rows_np(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-wise L2 normalize in NumPy (used after pooling)."""
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / (n + eps)

    # ---------------- Public: simple embed APIs ----------------

    def embed(self, img_np: np.ndarray) -> np.ndarray:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        feats = self._forward(batch)["global"]  # (1,D)
        return feats.squeeze(0).numpy()

    def embed_tokens(self, img_np: np.ndarray) -> np.ndarray:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        toks = self._forward(batch)["tokens"]   # (1,T,D)
        return toks.squeeze(0).numpy()

    def embed_both(self, img_np: np.ndarray) -> Dict[str, np.ndarray]:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        out = self._forward(batch)
        return {
            "global": out["global"].squeeze(0).numpy(),   # (D,)
            "tokens": out["tokens"].squeeze(0).numpy(),   # (T,D)
        }

    def embed_batch(self, imgs_np: Iterable[np.ndarray]) -> np.ndarray:
        """Global embeddings only (for quick indexing) -> (N,D)."""
        pils = [self._to_pil_rgb(a) for a in imgs_np]
        batch = self._prep_batch(pils)
        feats = self._forward(batch)["global"]  # (N,D)
        return feats.numpy()

    # ---------------- Rotation batching ----------------

    def _embed_rotations_np(
        self,
        img_np: np.ndarray,
        angles: Iterable[float],
        resample: int = Image.BILINEAR,
    ) -> Tuple[List[Image.Image], np.ndarray, np.ndarray]:
        """
        Rotate input once per angle, run forward in a single batch.
        Returns:
          - pils: rotated PIL images (for mask creation)
          - G: (A,D) globals
          - T: (A, H*W, D) tokens (row-normalized)
        """
        pil0 = self._to_pil_rgb(img_np)
        pils = [pil0.rotate(float(a), resample=resample, expand=False) for a in angles]
        batch = self._prep_batch(pils)
        out = self._forward(batch)
        G = out["global"].numpy()
        T = out["tokens"].numpy()
        return pils, G, T

    # ---------------- Public: query rotations ----------------

    def embed_query_rotations(
        self,
        img_np: np.ndarray,
        angles: Iterable[float] = (0.0, 90.0, 180.0, 270.0),
        resample: int = Image.BILINEAR,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Rotation augmentation only.
        Returns a list of dicts with angle, global, tokens (14×14).
        """
        pils, G, T = self._embed_rotations_np(img_np, angles, resample=resample)
        results: List[Dict[str, np.ndarray]] = []
        for i, a in enumerate(angles):
            results.append({
                "angle": float(a),
                "global": G[i].astype(np.float32).copy(),      # (D,)
                "tokens": T[i].astype(np.float32).copy(),      # (196,D) for ViT-B/16 224px
            })
        return results

    # ---------------- Mask & pooling helpers ----------------

    @staticmethod
    def _pil_to_mask_grid(pil_img: Image.Image, grid_hw: int, bg_threshold: float = 0.02) -> np.ndarray:
        """
        Downsample PIL grayscale to (grid_hw,grid_hw) and threshold to boolean foreground.
        """
        g = pil_img.convert("L").resize((grid_hw, grid_hw), resample=Image.BILINEAR)
        arr = np.asarray(g, dtype=np.float32) / 255.0
        return (arr >= float(bg_threshold))  # (grid_hw, grid_hw) bool

    @staticmethod
    def _bins(in_hw: int, out_hw: int) -> List[Tuple[int, int]]:
        """
        Adaptive-avg-pool-style integer binning boundaries (start, end) for each pooled cell.
        """
        import math
        bounds: List[Tuple[int, int]] = []
        for i in range(out_hw):
            start = math.floor((i * in_hw) / out_hw)
            end = math.floor(((i + 1) * in_hw) / out_hw)
            # guarantee at least one element
            end = max(end, start + 1)
            bounds.append((start, end))
        return bounds

    def _pool_tokens_and_mask(
        self,
        tokens14: np.ndarray,    # (196,D)
        qmask14_hw: np.ndarray,  # (14,14) bool
        out_hw: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive-average pooling for tokens and majority pooling for masks.
        Returns:
          - Qt: (out_hw*out_hw, D) row-normalized
          - qm: (out_hw*out_hw,) bool
        """
        ghw = int(self.grid_hw)
        if out_hw == ghw:
            Qt = tokens14.astype(np.float32, copy=True)
            qm = qmask14_hw.reshape(-1).copy()
            return Qt, qm

        x = tokens14.reshape(ghw, ghw, tokens14.shape[1])  # (14,14,D)
        out_tok = np.zeros((out_hw, out_hw, x.shape[2]), dtype=np.float32)

        rows = self._bins(ghw, out_hw)
        cols = self._bins(ghw, out_hw)
        for oi, (rs, re) in enumerate(rows):
            for oj, (cs, ce) in enumerate(cols):
                patch = x[rs:re, cs:ce, :]
                out_tok[oi, oj, :] = patch.mean(axis=(0, 1))

        Qt = out_tok.reshape(out_hw * out_hw, x.shape[2]).astype(np.float32)
        Qt = self._l2norm_rows_np(Qt)

        # mask: keep if >= 5% foreground within block
        m = qmask14_hw
        out_m = np.zeros((out_hw, out_hw), dtype=bool)
        for oi, (rs, re) in enumerate(rows):
            for oj, (cs, ce) in enumerate(cols):
                block = m[rs:re, cs:ce]
                out_m[oi, oj] = (block.mean() >= 0.05)

        qm = out_m.reshape(-1)
        return Qt, qm

    # ---------------- Public: query (angle × scale) augmentations ----------------

    def embed_query_augmentations(
        self,
        img_np: np.ndarray,
        *,
        angles: Iterable[float] = (0.0, 90.0, 180.0, 270.0),
        scales: Iterable[int] = (1, 2, 4, 8, 14),
        resample: int = Image.BILINEAR,
    ) -> List[Dict]:
        """
        Returns a list over rotations. Each item:
        {
          "angle": float,
          "global": (D,),
          "tokens14": (196,D),      # 14x14 row-normalized
          "qmask14": (196,) bool,   # 14x14 flattened
          "scales": [
            {"k": k, "Qt": (k*k,D), "qm": (k*k,) bool}  # row-normalized, pooled
            for k in scales
          ]
        }
        """
        pils, G, T = self._embed_rotations_np(img_np, angles, resample=resample)
        ghw = int(self.grid_hw)

        results: List[Dict] = []
        for i, a in enumerate(angles):
            tokens14 = T[i].astype(np.float32)                # (196,D), already L2-normalized
            qmask14_hw = self._pil_to_mask_grid(pils[i], ghw) # (14,14)
            qmask14 = qmask14_hw.reshape(-1)

            scales_list = []
            for k in scales:
                k = int(k)
                Qt, qm = self._pool_tokens_and_mask(tokens14, qmask14_hw, out_hw=k)
                scales_list.append({"k": k, "Qt": Qt, "qm": qm})

            results.append({
                "angle": float(a),
                "global": G[i].astype(np.float32).copy(),
                "tokens14": tokens14,
                "qmask14": qmask14,
                "scales": scales_list,
            })
        return results


# Public instance (stable import)
model = _HFVisionEncoder()
