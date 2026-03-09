#!/usr/bin/env python3
"""
Compute background vs foreground ratio for the Allen CCF (average mouse brain).

Two modes:
  A) 3D volume ratio (recommended): uses annotation volume (label==0 is background).
  B) 2D slice ratio estimate: sample random slice planes and compute fg fraction in each slice
     using include_annotation=True (labels > 0).

Examples:
  # 3D ratio using annotation labels (best for "Allen brain mask" definition)
  python -m scripts.allen_bg_fg_ratio --mode volume

  # 2D slice ratio distribution over 2000 random planes (unfiltered)
  python -m scripts.allen_bg_fg_ratio --mode slices --num-slices 2000

  # 2D slices but only keep "non-trivial" slices with >= 10% tissue
  python -m scripts.allen_bg_fg_ratio --mode slices --num-slices 2000 --min-fg-frac 0.10

  # Also compute intensity-threshold ratio (less robust; optional)
  python -m scripts.allen_bg_fg_ratio --mode volume --also-intensity --thr-pct 0.10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np

from allensdk.core.reference_space_cache import ReferenceSpaceCache

# Uses your helper for plane sampling (slices mode)
from volume.volume_helper import AllenVolume


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _random_unit_vector(rng: np.random.Generator) -> Tuple[float, float, float]:
    # Uniform on sphere: sample normal via Gaussian then normalize
    v = rng.normal(size=(3,))
    v = _unit(v)
    return (float(v[0]), float(v[1]), float(v[2]))


def _volume_ratios_from_labels(cache_dir: str, res_um: int) -> dict:
    cache_dir_p = Path(cache_dir)
    rc = ReferenceSpaceCache(
        resolution=int(res_um),
        reference_space_key="annotation/ccf_2017",
        manifest=cache_dir_p / "manifest.json",
    )

    labels_zyx, _ = rc.get_annotation_volume()  # (Z,Y,X) int labels
    labels_zyx = labels_zyx.astype(np.int32, copy=False)

    total = int(labels_zyx.size)
    bg = int(np.count_nonzero(labels_zyx == 0))
    fg = total - bg

    bg_frac = bg / total
    fg_frac = fg / total
    bg_over_fg = (bg / fg) if fg > 0 else float("inf")
    fg_over_bg = (fg / bg) if bg > 0 else float("inf")

    return {
        "total_voxels": total,
        "background_voxels": bg,
        "foreground_voxels": fg,
        "background_fraction": float(bg_frac),
        "foreground_fraction": float(fg_frac),
        "background_over_foreground": float(bg_over_fg),
        "foreground_over_background": float(fg_over_bg),
        "definition": "annotation labels: background=0, foreground>0",
    }


def _volume_ratios_from_intensity(cache_dir: str, res_um: int, thr_pct: float) -> dict:
    """
    Optional: define foreground by intensity thresholding the Allen template volume.
    This is generally less clean than using annotation labels, but can be reported as a secondary check.
    """
    cache_dir_p = Path(cache_dir)
    rc = ReferenceSpaceCache(
        resolution=int(res_um),
        reference_space_key="annotation/ccf_2017",
        manifest=cache_dir_p / "manifest.json",
    )
    vol_zyx, _ = rc.get_template_volume()  # float-ish
    V = vol_zyx.astype(np.float32, copy=False)

    lo, hi = np.percentile(V, (0.5, 99.5))
    thr = float(lo + float(thr_pct) * float(hi - lo))

    fg_mask = V > thr
    total = int(V.size)
    fg = int(np.count_nonzero(fg_mask))
    bg = total - fg

    bg_frac = bg / total
    fg_frac = fg / total
    bg_over_fg = (bg / fg) if fg > 0 else float("inf")
    fg_over_bg = (fg / bg) if bg > 0 else float("inf")

    return {
        "total_voxels": total,
        "background_voxels": bg,
        "foreground_voxels": fg,
        "background_fraction": float(bg_frac),
        "foreground_fraction": float(fg_frac),
        "background_over_foreground": float(bg_over_fg),
        "foreground_over_background": float(fg_over_bg),
        "threshold": thr,
        "lo_p0.5": float(lo),
        "hi_p99.5": float(hi),
        "thr_pct": float(thr_pct),
        "definition": "template intensity: foreground = I > (lo + thr_pct*(hi-lo)) using percentiles (0.5,99.5)",
    }


def _slice_ratio_stats(
    cache_dir: str,
    res_um: int,
    num_slices: int,
    size: int,
    pixel: float,
    seed: int,
    min_fg_frac: float,
) -> dict:
    """
    Estimate fg/bg ratios on *2D slice images* by sampling random planes and using annotation labels per slice.
    Foreground definition: labels > 0.
    """
    rng = np.random.default_rng(int(seed))

    allen = AllenVolume(cache_dir=cache_dir, resolution_um=res_um)
    Z, Y, X = allen.get_dimension()

    # Depth range: use half diagonal in voxel units to cover full volume.
    # This will include many empty slices; filter with min_fg_frac if desired.
    half_diag = 0.5 * math.sqrt((X - 1) ** 2 + (Y - 1) ** 2 + (Z - 1) ** 2)

    fg_fracs = []
    kept = 0
    tried = 0

    while kept < num_slices and tried < max(num_slices * 50, num_slices + 100):
        tried += 1
        n = _random_unit_vector(rng)
        depth = float(rng.uniform(-half_diag, half_diag))
        rot = float(rng.uniform(0.0, 360.0))

        sl = allen.get_slice(
            n,
            depth=depth,
            rotation=rot,
            size=int(size),
            pixel=float(pixel),
            include_annotation=True,
        )
        if sl.labels is None:
            continue
        fg = float(np.count_nonzero(sl.labels > 0))
        frac = fg / float(sl.labels.size)

        if frac >= float(min_fg_frac):
            fg_fracs.append(frac)
            kept += 1

    fg_fracs = np.asarray(fg_fracs, dtype=np.float64)
    if fg_fracs.size == 0:
        return {
            "num_requested": int(num_slices),
            "num_kept": 0,
            "num_tried": int(tried),
            "min_fg_frac_filter": float(min_fg_frac),
            "error": "No slices met the filter; lower --min-fg-frac or increase --num-slices.",
        }

    # Convert fg distribution to bg too
    bg_fracs = 1.0 - fg_fracs

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p))

    return {
        "definition": "per-slice labels: foreground = labels>0, background = labels==0",
        "num_requested": int(num_slices),
        "num_kept": int(fg_fracs.size),
        "num_tried": int(tried),
        "slice_size_px": int(size),
        "pixel_step_vox": float(pixel),
        "min_fg_frac_filter": float(min_fg_frac),
        "foreground_fraction": {
            "mean": float(fg_fracs.mean()),
            "median": float(np.median(fg_fracs)),
            "p05": q(fg_fracs, 0.05),
            "p25": q(fg_fracs, 0.25),
            "p75": q(fg_fracs, 0.75),
            "p95": q(fg_fracs, 0.95),
            "min": float(fg_fracs.min()),
            "max": float(fg_fracs.max()),
        },
        "background_fraction": {
            "mean": float(bg_fracs.mean()),
            "median": float(np.median(bg_fracs)),
            "p05": q(bg_fracs, 0.05),
            "p25": q(bg_fracs, 0.25),
            "p75": q(bg_fracs, 0.75),
            "p95": q(bg_fracs, 0.95),
            "min": float(bg_fracs.min()),
            "max": float(bg_fracs.max()),
        },
    }


def _print_block(title: str, d: dict) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in d.items():
        print(f"{k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Allen CCF background vs foreground ratio")
    ap.add_argument("--cache-dir", type=str, default="volume/data/allen")
    ap.add_argument("--res-um", type=int, default=25)

    ap.add_argument(
        "--mode",
        type=str,
        choices=["volume", "slices"],
        default="volume",
        help="volume: compute 3D ratio from annotation volume; slices: estimate 2D slice ratios.",
    )

    # Intensity mode (optional secondary statistic)
    ap.add_argument("--also-intensity", action="store_true", help="Also compute ratio by intensity threshold.")
    ap.add_argument("--thr-pct", type=float, default=0.10, help="Intensity threshold pct in [0,1]. Used if --also-intensity.")

    # Slices mode args
    ap.add_argument("--num-slices", type=int, default=1000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--pixel", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--min-fg-frac",
        type=float,
        default=0.0,
        help="In slices mode, discard slices with fg fraction below this (e.g., 0.10 keeps only tissue-heavy slices).",
    )

    args = ap.parse_args()

    # 1) Volume (annotation) — recommended for a clean "brain mask" definition
    if args.mode == "volume":
        r_lab = _volume_ratios_from_labels(args.cache_dir, args.res_um)
        _print_block("3D volume ratio (annotation labels)", r_lab)

        if args.also_intensity:
            r_int = _volume_ratios_from_intensity(args.cache_dir, args.res_um, args.thr_pct)
            _print_block("3D volume ratio (template intensity threshold) — secondary", r_int)

    # 2) Slice distribution — useful for retrieval papers
    else:
        r_sl = _slice_ratio_stats(
            cache_dir=args.cache_dir,
            res_um=args.res_um,
            num_slices=args.num_slices,
            size=args.size,
            pixel=args.pixel,
            seed=args.seed,
            min_fg_frac=args.min_fg_frac,
        )
        _print_block("2D slice ratio distribution (random planes)", r_sl)


if __name__ == "__main__":
    main()
