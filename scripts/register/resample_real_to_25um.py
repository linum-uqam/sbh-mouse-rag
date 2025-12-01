from pathlib import Path
import SimpleITK as sitk


def resample_to_spacing(img: sitk.Image, new_spacing):
    """
    Resample a 3D image to new_spacing (sx, sy, sz) in physical units
    while keeping the same physical FOV.
    """
    original_spacing = img.GetSpacing()  # (sx, sy, sz)
    original_size = img.GetSize()        # (nx, ny, nz)

    # Compute new size so physical size stays the same:
    new_size = [
        int(round(osz * (ospc / nspc)))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    # Identity transform (no rotation/translation, just resampling grid change)
    transform = sitk.Transform()

    resampled = sitk.Resample(
        img,
        new_size,
        transform,
        sitk.sitkLinear,               # interpolation
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0,                              # default pixel value
        img.GetPixelID(),
    )
    return resampled


def main():
    real_path = Path("volume/data/real/registered_brain_10um.nii.gz")
    out_path  = Path("volume/data/real/registered_brain_25um.nii.gz")

    print(f"Reading {real_path} ...")
    real_img = sitk.ReadImage(str(real_path))

    print("Original spacing:", real_img.GetSpacing())
    print("Original size:", real_img.GetSize())

    # Target spacing (match Allen 25 µm)
    new_spacing = (0.025, 0.025, 0.025)  # <-- if your units are mm
    # If your spacing is already in µm (10.0, 10.0, 10.0), use (25.0, 25.0, 25.0)

    real_25 = resample_to_spacing(real_img, new_spacing)

    print("New spacing:", real_25.GetSpacing())
    print("New size:", real_25.GetSize())

    print(f"Writing {out_path} ...")
    sitk.WriteImage(real_25, str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
