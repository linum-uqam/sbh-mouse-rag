# index/model/dino.py
from __future__ import annotations
from typing import Iterable, List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class _HFVisionEncoder:
    """
    Minimal DINOv3 wrapper (ViT-Base/16 by default) with:
      - embed(img_np) -> (D,)
      - embed_tokens(img_np) -> (T,D)  [patch tokens only; no CLS; T = H*W]
      - embed_both(img_np) -> {"global": (D,), "tokens": (T,D)}
      - embed_batch(imgs_np) -> (N,D)
      - embed_pil_batch(pils) -> (N,D)

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
        """
        (H,W) or (H,W,1/3) in [0,1] or uint8 -> PIL RGB.
        Grayscale inputs are kept as luminance and expanded to 3 channels.
        """
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
        return self.processor(images=pil_list, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.dtype
        )

    # ---------------- Token / Forward utils ----------------

    def _patch_tokens_only(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, L, D)
        return: (B, H*W, D) with ONLY patch tokens (drop CLS/extras)
        """
        B, L, D = last_hidden.shape
        need = self.grid_hw * self.grid_hw

        if L >= need + 1:
            patches = last_hidden[:, 1 : 1 + need, :]
        elif L == need:
            patches = last_hidden
        elif L > need:
            patches = last_hidden[:, L - need : L, :]
        else:
            raise ValueError(
                f"Not enough tokens: L={L}, need≥{need} (grid={self.grid_hw}x{self.grid_hw})"
            )

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

    def embed_pil_batch(self, pils: Iterable[Image.Image]) -> np.ndarray:
        """
        Global embeddings from a batch of PIL images -> (N,D).
        Useful when the caller already has PILs (e.g. rotations).
        """
        pil_list = list(pils)
        if not pil_list:
            return np.zeros((0, self.encoder.config.hidden_size), dtype=np.float32)
        batch = self._prep_batch(pil_list)
        feats = self._forward(batch)["global"]  # (N,D)
        return feats.numpy()


# Public instance (stable import)
model = _HFVisionEncoder()
