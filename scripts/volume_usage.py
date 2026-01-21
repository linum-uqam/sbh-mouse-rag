# scripts/volume_usage.py
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from volume.volume_helper import AllenVolume, NiftiVolume, AnnotationHelper, Slice


def _volume_diag_bound(volume_shape_zyx, spacing_zyx=None):
    """
    Returns an upper bound for the bidirectional Chamfer distance:
      max_distance <= 2 * diagonal(volume)
    in voxel units if spacing_zyx is None, else in physical units.
    """
    Z, Y, X = map(int, volume_shape_zyx)
    if spacing_zyx is None:
        Lx, Ly, Lz = (X - 1), (Y - 1), (Z - 1)
    else:
        sz, sy, sx = map(float, spacing_zyx)  # spacing_zyx = (sz, sy, sx)
        Lx, Ly, Lz = (X - 1) * sx, (Y - 1) * sy, (Z - 1) * sz
    D = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    return 2.0 * D


def _describe_slice_points(sl: Slice, name: str, grid: int = 16):
    pts_vox = sl.sample_points_xyz(grid=grid, physical=False)
    pts_phy = sl.sample_points_xyz(grid=grid, physical=True)

    vmin, vmax = pts_vox.min(axis=0), pts_vox.max(axis=0)
    pmin, pmax = pts_phy.min(axis=0), pts_phy.max(axis=0)

    print(f"\n[{name}]")
    print(f"  volume_shape_zyx : {sl.volume_shape_zyx}")
    print(f"  spacing_zyx      : {sl.spacing_zyx}   (Z,Y,X)")
    print(f"  size_px          : {sl.size_px}")
    print(f"  origin_px_in_plane: {sl.origin_px_in_plane}")
    print(f"  depth_vox        : {sl.depth_vox:.4f}")
    print(f"  normal_xyz_unit  : {sl.normal_xyz_unit}")
    print(f"  rotation_deg     : {sl.rotation_deg:.4f}")
    print(f"  pixel_step_vox   : {sl.pixel_step_vox:.4f}")

    print("  sampled point range (voxel XYZ):")
    print(f"    min={vmin}, max={vmax}")

    print("  sampled point range (physical XYZ):")
    print(f"    min={pmin}, max={pmax}")


def _check_spacing_mismatch(a: Slice, b: Slice, tol_ratio: float = 1.10):
    """
    Flags spacing mismatches likely to make physical distances meaningless.
    tol_ratio=1.10 => warns if any axis differs by >10%.
    """
    az, ay, ax = map(float, a.spacing_zyx)
    bz, by, bx = map(float, b.spacing_zyx)

    ratios = np.array([az / (bz + 1e-12), ay / (by + 1e-12), ax / (bx + 1e-12)], dtype=np.float64)
    inv = 1.0 / (ratios + 1e-12)
    worst = float(np.max(np.maximum(ratios, inv)))

    print("\n[spacing check]")
    print(f"  spacing a (Z,Y,X): {a.spacing_zyx}")
    print(f"  spacing b (Z,Y,X): {b.spacing_zyx}")
    print(f"  worst ratio (either direction): {worst:.3f}x")

    if worst > tol_ratio:
        print("  WARNING: spacing mismatch detected. "
              "Distances with physical=True will be hard to interpret.")
    else:
        print("  OK: spacings are comparable (within tolerance).")


def _distance_sanity(a: Slice, b: Slice, name: str):
    print(f"\n[{name}] distance sanity")

    # Identity checks
    d_aa_phys = Slice.distance(a, a, grid=64, trim=0.10, physical=True)
    d_bb_phys = Slice.distance(b, b, grid=64, trim=0.10, physical=True)
    d_aa_vox  = Slice.distance(a, a, grid=64, trim=0.10, physical=False)
    d_bb_vox  = Slice.distance(b, b, grid=64, trim=0.10, physical=False)

    print(f"  self-distance a (physical=True) : {d_aa_phys:.6f}  (should be ~0)")
    print(f"  self-distance b (physical=True) : {d_bb_phys:.6f}  (should be ~0)")
    print(f"  self-distance a (physical=False): {d_aa_vox:.6f}  (should be ~0)")
    print(f"  self-distance b (physical=False): {d_bb_vox:.6f}  (should be ~0)")

    # Cross distances
    d_phys = Slice.distance(a, b, grid=64, trim=0.10, physical=True)
    d_vox  = Slice.distance(a, b, grid=64, trim=0.10, physical=False)

    max_phys = _volume_diag_bound(a.volume_shape_zyx, spacing_zyx=a.spacing_zyx)
    max_vox  = _volume_diag_bound(a.volume_shape_zyx, spacing_zyx=None)

    print(f"\n  distance a↔b (physical=True) : {d_phys:.6f}  (max bound ≈ {max_phys:.3f})")
    print(f"  distance a↔b (physical=False): {d_vox:.6f}  (max bound ≈ {max_vox:.3f})")

    if max_phys > 0:
        print(f"  physical distance as % of bound: {100.0 * d_phys / max_phys:.2f}%")
    if max_vox > 0:
        print(f"  voxel distance as % of bound   : {100.0 * d_vox / max_vox:.2f}%")


def main():
    out_dir = Path("out/volume/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Allen slice with labels only
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
    s_allen = allen.get_slice((0, 0, 1), depth=0.0, rotation=25.0, size=512, include_annotation=True)
    s_allen.save(out_dir / "allen_labels.png", overlay="labels", title="Allen labels")
    s_allen.save(out_dir / "allen_overlay.png", overlay="image+labels", title="Allen image+labels")

    # 2) Real volume with Allen annotation overlay (assumes aligned to Allen space)
    real = NiftiVolume("volume/data/real/registered_brain_25um.nii.gz")
    annot = AnnotationHelper(cache_dir="volume/data/allen", resolution_um=25)
    s_real = real.get_slice((0, 0, 1), depth=0.0, rotation=25.0, size=512, include_annotation=True, annotation_helper=annot)
    s_real.save(out_dir / "real_overlay.png", overlay="image+labels", title="Real + Allen labels")

    # 3) Cropped slice keeps labels
    c = s_real.crop_norm(cx=0.25, cy=0.75, rw=0.4, rh=0.4)
    c.save(out_dir / "real_overlay_cropped.png", overlay="image+labels", title="Crop with labels")

    # 4) Distance demo: left vs right hemisphere crops (same plane)
    c_left = s_real.crop_norm(cx=0.25, cy=0.50, rw=0.30, rh=0.30)
    c_right = s_real.crop_norm(cx=0.75, cy=0.50, rw=0.30, rh=0.30)

    c_left.save(out_dir / "crop_left.png", overlay="image+labels", title="Left crop")
    c_right.save(out_dir / "crop_right.png", overlay="image+labels", title="Right crop")

    dist_lr, info = Slice.distance(
        c_left,
        c_right,
        grid=64,
        trim=0.10,
        physical=True,
        also_return_mirror_diagnostic=True,
    )

    print("\n--- Original outputs ---")
    print("distance (allen vs real):", Slice.distance(s_allen, s_real, grid=64, trim=0.10, physical=True))
    print("distance (left crop vs right crop):", dist_lr)
    print("mirror diagnostic (mirror candidate b across axis=2/X):", info["distance_mirror_b"])

    # -------- New unit mismatch diagnosis --------
    print("\n\n==================== Unit / mismatch diagnostics ====================")

    _describe_slice_points(s_allen, "Allen slice", grid=16)
    _describe_slice_points(s_real, "Real slice", grid=16)

    _check_spacing_mismatch(s_allen, s_real, tol_ratio=1.10)

    _distance_sanity(s_allen, s_real, "Allen vs Real")

    print("\n\n==================== (Optional) crop diagnostic ====================")
    _describe_slice_points(c_left, "Left crop", grid=16)
    _describe_slice_points(c_right, "Right crop", grid=16)
    _distance_sanity(c_left, c_right, "Left crop vs Right crop")

    # Visual summary figure for left/right
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(c_left.image, cmap="gray")
    ax1.set_title("Left crop")
    ax1.axis("off")

    ax2.imshow(c_right.image, cmap="gray")
    ax2.set_title("Right crop")
    ax2.axis("off")

    fig.suptitle(
        f"Strict distance = {dist_lr:.4f} | Mirror diagnostic = {info['distance_mirror_b']:.4f}\n"
        f"(Axis of symmetry is axis=2 (X).)"
    )

    fig.tight_layout()
    fig.savefig(out_dir / "distance_demo_left_vs_right.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
