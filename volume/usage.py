# volume/usage.py
from pathlib import Path
from volume.volume_helper import AllenVolume, NiftiVolume, slice_distance

def main():
    out = Path("volume/out")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Allen template
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
    print("Allen dims (Z,Y,X):", allen.get_dimension())
    a_slice = allen.get_slice((0,0,1), depth=0.0, rotation=25.0, size=512)
    a_slice.save(out / "allen_slice.png", title="Allen template slice")

    # 2) Your NIfTI volume
    real = NiftiVolume("volume/data/real/real_mouse_brain_ras_25um.nii.gz")
    print("Real dims (Z,Y,X):", real.get_dimension())
    r_slice = real.get_slice((0,0,1), depth=0.0, rotation=25.0, size=512)
    r_slice.save(out / "real_slice.png", title="Real volume slice")

    c_norm = r_slice.crop_norm(cx=0.25, cy=0.75, rw=0.4, rh=0.4)
    c_norm.save(out / "crop_norm.png")

    # Distances — now based on slices
    print("Allen–Allen (same):", slice_distance(a_slice, a_slice))
    print("Allen–Real (same pose):", slice_distance(a_slice, r_slice))

    # Different pose examples
    s1 = allen.get_slice((1,0,0), depth=20.0, size=512)
    s2 = allen.get_slice((0,1,0), depth=20.0, size=512)
    print("Orthogonal @ same depth:", slice_distance(s1, s2))

if __name__ == "__main__":
    main()