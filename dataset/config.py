from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    num_slices = number of base planes / full slices.

    Total rows generated:
        num_slices * (1 + 3 * len(crop_aspect_ratios))
    """

    out_dir: Path = Path("out/dataset/data")
    csv_path: Path = Path("out/dataset/dataset.csv")

    allen_cache_dir: str = "volume/data/allen"
    allen_res_um: int = 25
    real_volume_path: str = "volume/data/real/real_mouse_brain_ras_25um.nii.gz"

    num_slices: int = 1000
    slice_size: int = 512

    max_crop_attempts: int = 50
    min_crop_frac: float = 0.3
    max_crop_frac: float = 0.7

    crop_aspect_labels: tuple[str, ...] = ("square", "wide", "tall")
    crop_aspect_ratios: tuple[tuple[float, float], ...] = (
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0),
    )

    save_images: bool = True
    seed: int = 42
