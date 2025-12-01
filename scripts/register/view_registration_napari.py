import napari
import SimpleITK as sitk

allen_img = sitk.ReadImage("volume/data/allen/average_template_25.nrrd")
real_img  = sitk.ReadImage("volume/data/real/registered_brain_25um.nii.gz")


allen = sitk.GetArrayFromImage(allen_img)      # (Z,Y,X)
real  = sitk.GetArrayFromImage(real_img)

viewer = napari.Viewer()
viewer.add_image(allen, name="Allen", colormap="gray", blending="additive")
viewer.add_image(real,  name="Real registered", colormap="magenta", blending="additive", opacity=0.5)
napari.run()