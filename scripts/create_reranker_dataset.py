from __future__ import annotations

import argparse
from pathlib import Path

from dataset import DatasetConfig, MouseBrainDatasetBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create Allen/real dataset (full slices + crops) "
            "for reranker training/validation."
        )
    )

    # Output
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/reranker_dataset/data"),
        help="Directory for PNG images (default: out/reranker_dataset/data).",
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=Path("out/reranker_dataset/dataset.csv"),
        help="Path to CSV file (default: out/reranker_dataset/dataset.csv).",
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
        help=(
            "Number of base planes to sample. "
            "Final row count is larger because each plane generates one full slice "
            "plus multiple crops."
        ),
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
        default=50,
        help="Max random crop attempts per requested crop.",
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

    p.add_argument(
        "--crop-aspects",
        type=str,
        nargs="+",
        default=["square", "wide", "tall"],
        choices=["square", "wide", "tall"],
        help=(
            "Crop aspect families to generate. "
            "Default: square wide tall"
        ),
    )
    p.add_argument(
        "--wide-ratio",
        type=float,
        default=2.0,
        help="Width/height ratio used for 'wide' crops (default: 2.0).",
    )
    p.add_argument(
        "--tall-ratio",
        type=float,
        default=2.0,
        help="Height/width ratio used for 'tall' crops (default: 2.0).",
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
        default=123,
        help="Random seed.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    crop_aspect_ratios: tuple[tuple[float, float], ...] = tuple(
        {
            "square": (1.0, 1.0),
            "wide": (float(args.wide_ratio), 1.0),
            "tall": (1.0, float(args.tall_ratio)),
        }[name]
        for name in args.crop_aspects
    )

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
        crop_aspect_ratios=crop_aspect_ratios,
        save_images=not args.no_save_images,
        seed=args.seed,
    )

    builder = MouseBrainDatasetBuilder(cfg)
    builder.run()


if __name__ == "__main__":
    main()