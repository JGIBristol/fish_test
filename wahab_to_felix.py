"""
Script to convert Wahab's data to Felix's format

i.e. read in a stack of 2d tiffs, and save it as a 3d tiff

"""

import os
import pathlib
import multiprocessing
from typing import Iterable
import cv2
import tifffile
import numpy as np

# Directories holding stuff
IN_DIR = pathlib.Path(
    "/home/mh19137/zebrafish_rdsf/DATABASE/uCT/Wahab_clean_dataset/low_res_clean_v3/"
)
OUT_DIR = pathlib.Path(
    "/home/mh19137/zebrafish_rdsf/1Felix and Rich make models/wahabs_scans/"
)


def _input_ns(in_dir: pathlib.Path) -> Iterable[str]:
    """
    Get the valid numbers as a three-digit string

    """
    for path in in_dir.glob("[0-9]" * 3):
        if path.is_dir():
            yield path.name


def _3d_img(stack_dir: pathlib.Path) -> np.ndarray:
    """
    Read tiff stack into an array

    """
    # Find the individual images
    img_paths = sorted(list(stack_dir.glob("*.tiff")))

    # Init the array
    img0 = cv2.imread(str(img_paths[0]), cv2.IMREAD_GRAYSCALE)
    shape = img0.shape

    retval = np.empty((len(img_paths), *shape), dtype=np.uint8)
    retval[0] = img0

    for i, img_path in enumerate(img_paths[1:], start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        retval[i] = img

    return retval


def _out_path(n: str) -> pathlib.Path:
    return pathlib.Path(OUT_DIR) / f"{n}.tif"


def _write_tif(stack: np.ndarray, n: str) -> None:
    """
    Write the 3d array to a tiff

    """
    try:
        tifffile.imwrite(str(_out_path(n)), stack)
    except KeyboardInterrupt:
        print(f"Caught KeyboardInterrupt, exiting and deleting {_out_path(n)}")
        os.remove(_out_path(n))

    print("Wrote ", n)


def _make_3d_tiff(n: str) -> None:
    """
    Check if it already exists; if not write a 3d copy to the output directory

    """
    if _out_path(n).exists():
        print(f"Skipping {n}")
        return

    print(f"Reading {n}")
    array = _3d_img(IN_DIR / n / "reconstructed_tifs")

    _write_tif(array, n)


def main():
    """
    Read in the stack of .tiff images, write them out to a 3d

    """

    with multiprocessing.Pool(processes=12) as pool:
        pool.map(_make_3d_tiff, _input_ns(IN_DIR))


if __name__ == "__main__":
    main()
