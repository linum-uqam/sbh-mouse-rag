from pathlib import Path
from volume.volume_helper import AllenVolume, NiftiVolume, AnnotationHelper, Slice

out_dir = Path("out/volume/"); out_dir.mkdir(parents=True, exist_ok=True)

# 1) Allen slice with labels only
allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
s_allen = allen.get_slice((0,0,1), depth=0.0, rotation=25.0, size=512, include_annotation=True)
s_allen.save(out_dir / "allen_labels.png", overlay="labels", title="Allen labels")
s_allen.save(out_dir / "allen_overlay.png", overlay="image+labels", title="Allen image+labels")

# 2) Real volume with Allen annotation overlay (assumes aligned to Allen space)
# real = NiftiVolume("volume/data/real/real_mouse_brain_ras_25um.nii.gz")
real = NiftiVolume("volume/data/real/registered_brain_25um.nii.gz")
annot = AnnotationHelper(cache_dir="volume/data/allen", resolution_um=25)
s_real = real.get_slice((0,0,1), depth=0.0, rotation=25.0, size=512, include_annotation=True, annotation_helper=annot)
s_real.save(out_dir / "real_overlay.png", overlay="image+labels", title="Real + Allen labels")

# 3) Cropped slice keeps labels
c = s_real.crop_norm(cx=0.25, cy=0.75, rw=0.4, rh=0.4)
c.save(out_dir / "real_overlay_cropped.png", overlay="image+labels", title="Crop with labels")

# 4) Distance still works (independent of labels)
print("distance:", Slice.distance(s_allen, s_real))
