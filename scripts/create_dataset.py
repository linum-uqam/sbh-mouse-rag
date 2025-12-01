# scripts/create_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path

# Use the public API re-exported in dataset/__init__.py
from dataset import DatasetConfig, MouseBrainDatasetBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Allen/real dataset with full slices + random crops."
    )

    # Output
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/dataset/data"),
        help="Directory for PNG images (default: dataset/data).",
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=Path("out/dataset/dataset.csv"),
        help="Path to CSV file (default: dataset/dataset.csv).",
    )

    # Volumes
    p.add_argument(
        "--allen-cache-dir",
        type=str,
        default="volume/data/allen",
        help="Directory where Allen SDK cache/manifest is stored.",
    )
    p.add_argument(
        "--allen-res-um",
        type=int,
        default=25,
        help="Allen template resolution in microns (passed to AllenVolume).",
    )
    p.add_argument(
        "--real-volume-path",
        type=str,
        default="volume/data/real/real_mouse_brain_ras_25um.nii.gz",
        help="Path to real mouse brain NIfTI volume.",
    )

    # Sampling
    p.add_argument(
        "--num-slices",
        type=int,
        default=1000,
        help="Total number of dataset rows (full + crops).",
    )
    p.add_argument(
        "--slice-size",
        type=int,
        default=512,
        help="Size of extracted slices (square).",
    )

    # Crops
    p.add_argument(
        "--max-crop-attempts",
        type=int,
        default=4,
        help="Max random crop attempts per valid full slice.",
    )
    p.add_argument(
        "--min-crop-frac",
        type=float,
        default=0.3,
        help="Min crop size as fraction of slice (0–1).",
    )
    p.add_argument(
        "--max-crop-frac",
        type=float,
        default=0.7,
        help="Max crop size as fraction of slice (0–1).",
    )

    # Misc
    p.add_argument(
        "--no-save-images",
        action="store_true",
        help="If set, do NOT write PNG images (CSV only).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DatasetConfig(
        out_dir=args.out_dir,
        csv_path=args.csv_path,
        allen_cache_dir=args.allen_cache_dir,
        allen_res_um=args.allen_res_um,
        real_volume_path=args.real_volume_path,
        num_slices=args.num_slices,
        slice_size=args.slice_size,
        max_crop_attempts=args.max_crop_attempts,
        min_crop_frac=args.min_crop_frac,
        max_crop_frac=args.max_crop_frac,
        save_images=not args.no_save_images,
        seed=args.seed,
    )

    builder = MouseBrainDatasetBuilder(cfg)
    builder.run()


if __name__ == "__main__":
    main()
