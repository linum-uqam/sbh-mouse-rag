# scripts/train_reranker.py
from __future__ import annotations
from pathlib import Path
import argparse

from index.reranker import TrainingConfig, TrainingRun
from index.utils import log  # or replace with print


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train two-tower reranker (MSE regression) on image pairs."
    )

    # --- data selection ---
    p.add_argument(
        "--data-mode",
        type=str,
        default="auto",
        choices=["auto", "csv", "volume"],
        help="How to obtain training pairs: 'csv', 'volume', or 'auto' (default).",
    )

    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to CSV with 'query_path', 'candidate_path', 'target' columns (for CSV mode).",
    )

    p.add_argument(
        "--allen-cache-dir",
        type=str,
        default="volume/data/allen",
        help="Allen cache directory (for volume mode).",
    )
    p.add_argument(
        "--allen-resolution",
        type=int,
        default=25,
        help="Allen resolution in microns (for volume mode).",
    )
    p.add_argument(
        "--real-nifti",
        type=str,
        default="",
        help="Path to registered real brain NIfTI (for volume mode).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Number of pair samples to draw from volumes (for volume mode).",
    )
    p.add_argument(
        "--slice-size",
        type=int,
        default=224,
        help="Slice size (pixels) for volume mode.",
    )

    # --- training hyperparams ---
    p.add_argument(
        "--out",
        type=str,
        default="out/reranker/reranker.pt",
        help="Output path for trained reranker weights.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (CSV mode only; volume mode uses 0).",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data used for training.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data used for validation (rest is test).",
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

    csv_path = args.csv if args.csv else None
    real_nifti = args.real_nifti if args.real_nifti else None

    cfg = TrainingConfig(
        data_mode=args.data_mode,
        csv_path=csv_path,
        allen_cache_dir=args.allen_cache_dir,
        allen_resolution_um=args.allen_resolution,
        real_nifti_path=real_nifti,
        n_samples=args.n_samples,
        slice_size=args.slice_size,
        out_path=args.out,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    log("Reranker training config:")
    for k, v in cfg.to_dict().items():
        log(f"  {k:20s} = {v}")

    run = TrainingRun(cfg)
    metrics = run.run()

    log("Reranker training finished.")
    for k, v in metrics.items():
        log(f"  {k:15s} = {v:.6f}")


if __name__ == "__main__":
    main()
