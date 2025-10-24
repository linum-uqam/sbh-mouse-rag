from volume.volume_helper import AllenVolume, NiftiVolume 
import random
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

out = Path("dataset/data")
out.mkdir(parents=True, exist_ok=True)

def save_img(path: str | Path, arr): 
    plt.imsave(path, arr, cmap="gray") 
    print(f"Image saved as {path}")

def write_csv(csv_path: str, line):
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(line)
        
def is_valid(a_slice):
    ratio_threshold = 0.1
    value_threshold = 0.1
    mask = a_slice.image > value_threshold
    count = int(np.sum(mask))
    total = a_slice.image.size
    return (count / total) > ratio_threshold

stats={}
def create_dataset():
    save_img_flag=True
    number_sample_required=1000 # to put in args
    stats["number_sample"]=0
    stats["skipped"]=0
    # load volumes 
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
    real = NiftiVolume("volume/data/real/real_mouse_brain_ras_25um.nii.gz")

    headers=["allen_path", "real_path", "vector", "depth", "rotation"] 
    write_csv(Path("dataset/dataset.csv"), headers)

    while stats["number_sample"] < number_sample_required:

        # Create random slice values 
        v=[random.random(),random.random(),random.random()]
        depth=random.random()*512-256
        rotation=random.random()*360

        # Create slices 
        a_slice = allen.get_slice(v, depth, rotation, size=512)
        r_slice = real.get_slice(v, depth, rotation, size=512)

        # Save img if valid
        if (is_valid(a_slice)): 
            
            # Optionnal, save images
            if save_img_flag:
                allen_path= out / Path(f"{stats['number_sample']}_a_slice.png")
                save_img(allen_path, a_slice.image)
                real_path= out / Path(f"{stats['number_sample']}_r_slice.png")
                save_img(real_path, r_slice.image)            

            # Save row in dataset            
            write_csv(
                Path("dataset/dataset.csv"), 
                [
                    allen_path, real_path,
                    v, depth, rotation
                ])

            stats["number_sample"]+=1
        else:
            stats["skipped"]+=1
        
def run():
    random.seed(42)
    create_dataset()

if __name__=="__main__":
    run()