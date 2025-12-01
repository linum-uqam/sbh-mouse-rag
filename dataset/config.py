# dataset/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    Configuration for dataset generation.

    num_slices is the TOTAL number of rows in the CSV (full + crops).
    """

    # Output paths
    out_dir: Path = Path("out/dataset/data")
    csv_path: Path = Path("out/dataset/dataset.csv")

    # Volumes
    allen_cache_dir: str = "volume/data/allen"
    allen_res_um: int = 25
    real_volume_path: str = "volume/data/real/real_mouse_brain_ras_25um.nii.gz"

    # Sampling
    num_slices: int = 1000      # total rows (full + crops)
    slice_size: int = 512

    # Crops per plane
    max_crop_attempts: int = 4
    min_crop_frac: float = 0.3  # min relative crop size
    max_crop_frac: float = 0.7  # max relative crop size

    # Misc
    save_images: bool = True
    seed: int = 42
