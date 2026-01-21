# index/utils.py
from __future__ import annotations
from typing import Iterable
from pathlib import Path

from PIL import Image
import numpy as np


def log(title: str, lines: Iterable[str] | None = None) -> None:
    """
    Minimal standardized logger.
      log("tokens", ["training...", "nlist: 2048"])
      log("slices", ["Saved: path/to/file"])
      log("step", None)  # just a header line
    """
    if title and len(title)!=0 : 
        print(f"\n[{title}]")
    if lines is None:
        return
    for s in lines:
        print(f"  {s}")

# ---------------------------- IO ----------------------------------

def load_image_gray(path: str | Path) -> np.ndarray:
  """
  Load an image from disk as grayscale [0,1].
  """
  p = Path(path)
  if not p.exists():
      raise FileNotFoundError(f"Query image not found: {p}")
  im = Image.open(p).convert("L")
  arr = np.array(im, dtype=np.float32)
  if arr.max() > 1.0:
      arr = arr / 255.0
  return np.clip(arr, 0.0, 1.0)


