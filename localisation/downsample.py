"""
The images as they exist on the RDSF are larger than I would like (it takes a while to operate on them),
so here I will downsample them to a more manageable size.

"""

import os
import sys
import pathlib
from multiprocessing import Pool

import numpy as np
from skimage.transform import resize

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dev import image_io, util


def downsample(img_dir: pathlib.Path) -> None:
    """
    Downsample an image and save it to disk

    """
    # Extract the image number
    img_no = img_dir.name
    path = f"{util.config()['downsampled_img_dir']}{img_no}.npy"

    if os.path.exists(path):
        print(f"Skipping {path}")
        return

    # Read all the tifs into a numpy array
    print(f"Reading {img_no}")
    img_arr = image_io.read_tiffstack(int(img_no))

    # Downsample this array
    rescaled_size = (512, 512, 512)
    print(f"Downscaling {img_no}")
    img_arr = resize(img_arr, rescaled_size, order=1)

    # Save it to disk
    np.save(path, img_arr)
    print(f"Saved {path}")


def main():
    """
    Downsample some images

    """
    img_dir = (
        pathlib.Path(image_io.user_config()["rdsf_dir"])
        / util.config()["wahab_data_dir"]
    )

    dirs = list(img_dir.glob(r"[0-9][0-9][0-9]"))

    if not os.path.exists(util.config()["downsampled_img_dir"]):
        os.makedirs(util.config()["downsampled_img_dir"], exist_ok=True)

    # Only choose a few dirs for now
    gen = np.random.default_rng(seed=0)
    dirs = gen.choice(dirs, size=12)

    with Pool() as pool:
        pool.map(downsample, dirs)


if __name__ == "__main__":
    main()
