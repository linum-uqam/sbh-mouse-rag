# index/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

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

def log(title: str, lines: Iterable[str] | None = None) -> None:
    """
    Minimal standardized logger.
      log("tokens", ["training...", "nlist: 2048"])
      log("slices", ["Saved: path/to/file"])
      log("step", None)  # just a header line
    """
    print(f"\n[{title}]")
    if lines is None:
        return
    for s in lines:
        print(f"  {s}")

def l2norm_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-12)

