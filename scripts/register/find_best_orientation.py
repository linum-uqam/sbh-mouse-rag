# scripts/find_best_orientation.py
from __future__ import annotations

import argparse
from itertools import permutations, product
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Brute-force search over axis permutations + flips to roughly "
            "align REAL brain volume orientation to the Allen template. "
            "Outputs a pre-oriented REAL volume."
        )
    )
    p.add_argument(
        "--allen",
        type=Path,
        required=True,
        help="Allen average brain at 25um (e.g. average_template_25.nrrd)",
    )
    p.add_argument(
        "--real",
        type=Path,
        required=True,
        help="REAL brain at 25um (e.g. registered_brain_25um.nii.gz)",
    )
    p.add_argument(
        "--out-real",
        type=Path,
        required=True,
        help="Output REAL volume with best orientation applied.",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=4,
        help="Subsampling factor for scoring (default: 4 → use every 4th voxel).",
    )
    return p.parse_args()


# ---------- Scoring helpers ----------

def _crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop two volumes to their common minimum shape."""
    min_z = min(a.shape[0], b.shape[0])
    min_y = min(a.shape[1], b.shape[1])
    min_x = min(a.shape[2], b.shape[2])
    return a[:min_z, :min_y, :min_x], b[:min_z, :min_y, :min_x]


def _normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32, copy=False)
    mean = float(vol.mean())
    std = float(vol.std())
    if std < 1e-6:
        return vol * 0.0
    return (vol - mean) / std


def corr_score(
    allen: np.ndarray,
    cand: np.ndarray,
    subsample: int = 4,
) -> float:
    """
    Compute a simple normalized correlation between two volumes.

    - Crops to common shape.
    - Optionally subsamples (e.g. step=4) to speed up.
    """
    a, b = _crop_to_common(allen, cand)

    if subsample > 1:
        a = a[::subsample, ::subsample, ::subsample]
        b = b[::subsample, ::subsample, ::subsample]

    a = _normalize(a).ravel()
    b = _normalize(b).ravel()

    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return num / den


# ---------- Main logic ----------

def main() -> None:
    args = parse_args()

    print(f"Reading Allen volume: {args.allen}")
    allen_img = sitk.ReadImage(str(args.allen))
    allen_arr = sitk.GetArrayFromImage(allen_img)  # (Z, Y, X)

    print(f"Reading REAL volume: {args.real}")
    real_img = sitk.ReadImage(str(args.real))
    real_arr = sitk.GetArrayFromImage(real_img)    # (Z, Y, X)

    print("Allen shape:", allen_arr.shape, "| REAL shape:", real_arr.shape)
    print("Starting orientation search (6 permutations × 8 flips = 48 candidates)...")

    best_score = -1e9
    best_perm = None
    best_flip = None

    # Try all permutations of axes (0,1,2) and all flip combinations.
    for perm in permutations((0, 1, 2)):
        real_perm = np.transpose(real_arr, axes=perm)
        for flip in product((False, True), repeat=3):
            cand = real_perm
            for axis, do_flip in enumerate(flip):
                if do_flip:
                    cand = np.flip(cand, axis=axis)

            score = corr_score(allen_arr, cand, subsample=args.subsample)

            print(
                f"perm={perm}, flip={flip} -> score={score:.4f}",
                flush=True,
            )

            if score > best_score:
                best_score = score
                best_perm = perm
                best_flip = flip

    print("\nBest orientation found:")
    print(f"  perm = {best_perm}")
    print(f"  flip = {best_flip}")
    print(f"  corr = {best_score:.4f}")

    # ---------- Apply best orientation to full REAL volume ----------

    # Apply perm + flip to full-res numpy array
    real_oriented = np.transpose(real_arr, axes=best_perm)
    for axis, do_flip in enumerate(best_flip):
        if do_flip:
            real_oriented = np.flip(real_oriented, axis=axis)

    # Build output SimpleITK image from oriented array.
    # Note: GetImageFromArray assumes array is (Z, Y, X).
    out_img = sitk.GetImageFromArray(real_oriented.astype(real_arr.dtype, copy=False))

    # Compute new spacing for the oriented image.
    # Original spacing from REAL image is in (sx, sy, sz) = (X, Y, Z).
    sx, sy, sz = real_img.GetSpacing()

    # Convert to array-axis order (Z, Y, X)
    spacing_arr = np.array([sz, sy, sx], dtype=float)
    spacing_arr_perm = spacing_arr[list(best_perm)]  # permuted for new axes

    # Map back: SimpleITK expects (X, Y, Z) = (axis2, axis1, axis0) of the array
    sx_new = float(spacing_arr_perm[2])
    sy_new = float(spacing_arr_perm[1])
    sz_new = float(spacing_arr_perm[0])

    out_img.SetSpacing((sx_new, sy_new, sz_new))

    # You can choose what to do with origin/direction; simplest is to keep REAL's.
    out_img.SetOrigin(real_img.GetOrigin())
    out_img.SetDirection(real_img.GetDirection())

    print(f"Writing oriented REAL volume to: {args.out_real}")
    sitk.WriteImage(out_img, str(args.out_real))
    print("Done.")


if __name__ == "__main__":
    main()
