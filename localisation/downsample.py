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


def downsample(img_no: int) -> None:
    """
    Downsample an image and save it to disk

    """
    #


def main():
    """
    Downsample some images

    """
    conf = util.config()
    img_dir = pathlib.Path(image_io.user_config()["rdsf_dir"]) / conf["wahab_data_dir"]

    dirs = list(img_dir.glob(r"[0-9][0-9][0-9]"))

    # Only choose a few dirs for now
    gen = np.random.default_rng(seed=0)
    dirs = gen.choice(dirs, size=5)
    for dir in dirs:
        print(dir)


if __name__ == "__main__":
    main()
