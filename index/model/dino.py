# index/model/dino.py
from __future__ import annotations

import os
from typing import Iterable, List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoModel


class _HFVisionEncoder:

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        self.model_id = model_id
        self.hf_token = os.getenv("HF_TOKEN")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if not self.hf_token:
            print(
                "[DINO] Warning: HF_TOKEN is not set. "
                "Gated model loading may fail."
            )

        self.encoder = AutoModel.from_pretrained(
            model_id,
            token=self.hf_token,
            torch_dtype=self.dtype,
        )
        self.encoder.to(self.device).eval()

        self.image_size = int(getattr(self.encoder.config, "image_size", 224))
        self.patch_size = int(getattr(self.encoder.config, "patch_size", 16))
        self.grid_hw = self.image_size // self.patch_size

        print(
            f"Using {model_id} on {self.device} | "
            f"img={self.image_size}, patch={self.patch_size}, grid={self.grid_hw}x{self.grid_hw}"
        )

    # ---------------- Image I/O ----------------

    @staticmethod
    def _to_pil_rgb(img_np: np.ndarray) -> Image.Image:
        if img_np.ndim == 2:
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

    def _preprocess_pil(self, pil: Image.Image) -> torch.Tensor:
        """
        Manual preprocessing:
          - RGB
          - resize shortest side to image_size
          - center crop image_size x image_size
          - [0,1] float
          - normalize with ImageNet stats
        Returns: (3,H,W) tensor on CPU float32
        """
        pil = pil.convert("RGB")

        w, h = pil.size
        target = self.image_size

        # Resize shortest side to target while keeping aspect ratio
        if w < h:
            new_w = target
            new_h = int(round(h * (target / w)))
        else:
            new_h = target
            new_w = int(round(w * (target / h)))

        pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)

        # Center crop
        left = max(0, (new_w - target) // 2)
        top = max(0, (new_h - target) // 2)
        pil = pil.crop((left, top, left + target, top + target))

        arr = np.asarray(pil).astype(np.float32) / 255.0  # (H,W,3)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)

        mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        x = (x - mean) / std
        return x

    def _prep_batch(self, pil_list: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self._preprocess_pil(p) for p in pil_list], dim=0)
        return batch.to(self.device, dtype=self.dtype)

    # ---------------- Token utils ----------------

    def _patch_tokens_only(self, last_hidden: torch.Tensor) -> torch.Tensor:
        B, L, D = last_hidden.shape
        need = self.grid_hw * self.grid_hw

        if L >= need + 1:
            patches = last_hidden[:, 1 : 1 + need, :]
        elif L == need:
            patches = last_hidden
        elif L > need:
            patches = last_hidden[:, L - need : L, :]
        else:
            raise ValueError(f"Not enough tokens: L={L}, need≥{need} (grid={self.grid_hw}x{self.grid_hw})")

        if patches.shape[1] != need:
            raise ValueError(f"Token trim mismatch: got {patches.shape[1]} expected {need}")
        return patches

    def _global_from_hidden(self, last_hidden: torch.Tensor) -> torch.Tensor:
        B, L, D = last_hidden.shape
        need = self.grid_hw * self.grid_hw
        if L >= need + 1:
            return last_hidden[:, 0, :]
        patches = self._patch_tokens_only(last_hidden)
        return patches.mean(dim=1)

    @torch.inference_mode()
    def _forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(pixel_values=pixel_values)
        hidden = out.last_hidden_state

        g = self._global_from_hidden(hidden)
        p = self._patch_tokens_only(hidden)

        g = F.normalize(g, p=2, dim=-1).to(dtype=torch.float32).cpu()
        p = F.normalize(p, p=2, dim=-1).to(dtype=torch.float32).cpu()
        return {"global": g, "tokens": p}

    # ---------------- Public: simple embed APIs ----------------

    def embed(self, img_np: np.ndarray) -> np.ndarray:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        feats = self._forward(batch)["global"]
        return feats.squeeze(0).numpy()

    def embed_tokens(self, img_np: np.ndarray) -> np.ndarray:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        toks = self._forward(batch)["tokens"]
        return toks.squeeze(0).numpy()

    def embed_both(self, img_np: np.ndarray) -> Dict[str, np.ndarray]:
        pil = self._to_pil_rgb(img_np)
        batch = self._prep_batch([pil])
        out = self._forward(batch)
        return {
            "global": out["global"].squeeze(0).numpy(),
            "tokens": out["tokens"].squeeze(0).numpy(),
        }

    def embed_batch(self, imgs_np: Iterable[np.ndarray]) -> np.ndarray:
        pils = [self._to_pil_rgb(a) for a in imgs_np]
        if not pils:
            return np.zeros((0, int(self.encoder.config.hidden_size)), dtype=np.float32)
        batch = self._prep_batch(pils)
        feats = self._forward(batch)["global"]
        return feats.numpy()

    def embed_pil_batch(self, pils: Iterable[Image.Image]) -> np.ndarray:
        pil_list = list(pils)
        if not pil_list:
            return np.zeros((0, int(self.encoder.config.hidden_size)), dtype=np.float32)
        batch = self._prep_batch(pil_list)
        feats = self._forward(batch)["global"]
        return feats.numpy()

    @torch.inference_mode()
    def embed_tokens_batch(
        self,
        imgs_np: Iterable[np.ndarray],
        *,
        pool: int = 2,
        out_dtype: np.dtype = np.float16,
    ) -> np.ndarray:
        pils = [self._to_pil_rgb(a) for a in imgs_np]
        if not pils:
            return np.zeros((0, 0, int(self.encoder.config.hidden_size)), dtype=out_dtype)

        pixel_values = self._prep_batch(pils)
        out = self.encoder(pixel_values=pixel_values)
        hidden = out.last_hidden_state

        tok = self._patch_tokens_only(hidden)
        B, N, D = tok.shape

        H = W = self.grid_hw
        if N != H * W:
            raise ValueError(f"Unexpected token count N={N} vs grid {H}x{W}")

        pool = 1 if pool is None else int(pool)
        if pool < 1:
            raise ValueError("pool must be >= 1")

        if pool > 1:
            tok_2d = tok.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            tok_2d = F.avg_pool2d(tok_2d, kernel_size=pool, stride=pool)
            h2, w2 = int(tok_2d.shape[2]), int(tok_2d.shape[3])
            tok = tok_2d.permute(0, 2, 3, 1).contiguous().reshape(B, h2 * w2, D)

        tok = F.normalize(tok, p=2, dim=-1)
        tok = tok.to(dtype=torch.float32).cpu().numpy()
        if out_dtype is not None:
            tok = tok.astype(out_dtype, copy=False)
        return tok


model = _HFVisionEncoder()