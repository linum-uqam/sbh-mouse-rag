# index/utils.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

__all__ = [
    "make_slice_id",
    "load_image",
    "image_to_token_mask",
]


def make_slice_id(ni: int, di: int, ri: int) -> int:
    """Compose a stable 64-bit integer id from (normal_idx, depth_idx, rot_idx)."""
    return (int(ni) << 32) | (int(di) << 16) | int(ri)


def load_image(path: Path) -> np.ndarray:
    """Load a grayscale image as float32 in [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def image_to_token_mask(img01: np.ndarray, grid_hw: int, bg_threshold: float = 0.02) -> np.ndarray:
    """
    Build a token-level foreground mask from the grayscale image.
    Returns (grid_hw, grid_hw) bool where True=foreground (mean intensity >= threshold).
    """
    pil = Image.fromarray((img01 * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    small = pil.resize((grid_hw, grid_hw), resample=Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    return (arr >= float(bg_threshold))
