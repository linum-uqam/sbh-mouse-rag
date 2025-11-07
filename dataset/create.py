from volume.volume_helper import AllenVolume, NiftiVolume
import random
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 

out = Path("dataset/data")
out.mkdir(parents=True, exist_ok=True)

def save_img(path: str | Path, arr):
    plt.imsave(path, arr, cmap="gray")

def write_csv(csv_path: Path, line):
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

stats = {}

def create_dataset():
    save_img_flag = True
    number_sample_required = 1000  # to put in args
    stats["number_sample"] = 0
    stats["skipped"] = 0

    # load volumes
    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
    real = NiftiVolume("volume/data/real/real_mouse_brain_ras_25um.nii.gz")

    csv_path = Path("dataset/dataset.csv")
    headers = ["allen_path", "real_path", "vector", "depth", "rotation"]
    write_csv(Path("dataset/dataset.csv"), headers)

    attempts = 0
    pbar = tqdm(total=number_sample_required, desc="Generating slices", dynamic_ncols=True,leave=True)


    while stats["number_sample"] < number_sample_required:
        attempts += 1

        # Create random slice values
        v = [random.random(), random.random(), random.random()]
        depth = random.random() * 512 - 256
        rotation = random.random() * 360

        # Create slices
        a_slice = allen.get_slice(v, depth, rotation, size=512)
        r_slice = real.get_slice(v, depth, rotation, size=512)

        # Validate using Allen mask (or swap to local is_valid if you prefer)
        if allen.is_valid_slice(a_slice):
            # Optional, save images
            if save_img_flag:
                allen_path = out / f"{stats['number_sample']}_a_slice.png"
                save_img(allen_path, a_slice.image)
                real_path = out / f"{stats['number_sample']}_r_slice.png"
                save_img(real_path, r_slice.image)
            else:
                allen_path = out / f"{stats['number_sample']}_a_slice.png"
                real_path = out / f"{stats['number_sample']}_r_slice.png"

            # Save row in dataset
            write_csv(
                csv_path,
                [
                    str(allen_path),
                    str(real_path),
                    v,
                    depth,
                    rotation,
                ],
            )

            stats["number_sample"] += 1
            pbar.update(1)
        else:
            stats["skipped"] += 1

        # live stats on the bar
        pbar.set_postfix(
            collected=stats["number_sample"],
            skipped=stats["skipped"],
            attempts=attempts,
            hit_rate=f"{(stats['number_sample']/max(1,attempts))*100:.1f}%"
        )
    pbar.close()

def run():
    random.seed(42)
    create_dataset()

if __name__ == "__main__":
    run()
