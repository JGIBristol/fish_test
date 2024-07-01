"""
Cache the images to disk as numpy arrays

"""

import os
import argparse

import numpy as np

from dev import image_io, util


def main(*, img_n: int, n_jobs: int):
    """Read the selected image"""

    cache_dir = util.config()["img_cache"]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    out_file = os.path.join(cache_dir, f"{img_n}.npy")
    if not os.path.exists(out_file):
        arr = image_io.read_tiffstack(img_n, n_jobs=n_jobs)
        np.save(out_file, arr)
    else:
        print(f"File {out_file} already exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("img_n", type=int, help="The image number")
    parser.add_argument(
        "--n_jobs",
        "-n",
        type=int,
        default=None,
        help="Number of jobs. Defaults to none - i.e. don't use threads",
    )
    main(**vars(parser.parse_args()))
