"""
Crop arrays according to the centres in the mastersheet

"""

import os
from multiprocessing import Pool

import numpy as np
import cv2

from dev import metadata, image_io


def _cropped_img(img_n: str, region: str) -> np.ndarray:
    """
    Read in an image and crop it around a central point

    """
    window_size = 160
    cropped = np.empty((window_size, window_size, window_size))

    img_dir = image_io.img_dir(img_n)

    # Parse the coordinates
    z, x, y = image_io.parse_roi(region)

    # Choose which images to read
    start_i = z - window_size // 2

    # Read these images and copy the right pixels to the out array
    for i in range(window_size):
        img_path = f"{img_dir}/{img_n:03}_{start_i + i:04}.tiff"
        assert os.path.exists(img_path), img_path

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        cropped[i] = img[
            img.shape[1] - y - window_size // 2 : img.shape[1] - y + window_size // 2,
            x - window_size // 2 : x + window_size // 2,
        ]

    return cropped


def out_dir() -> str:
    return "cropped_arrs/"


def _save(n: int, jaw_centre: str) -> None:
    """
    Save the cropped image as a numpy array
    """
    if not image_io.img_dir(n).is_dir():
        print(f"Skipping {n} - does not exist")
        return

    out_path = f"{out_dir()}/{n:03}.npy"
    if not os.path.exists(out_path):
        try:
            cropped = _cropped_img(n, jaw_centre)
        except Exception as e:
            print(f"Error with {n}: {e}")
            return
        np.save(out_path, cropped)

    else:
        print(f"Skipping {n}")


def main():
    """
    For each stack of tifs, read the mastersheet to find the jaw centre, read the right images and crop them
    Then save them as numpy arrays

    """
    # Read the relevant columns in the mastersheet
    mastersheet = metadata.mastersheet()[["old_n", "jaw_center"]]

    # Choose a directory for images to be saved
    if not os.path.exists(out_dir()):
        os.mkdir(out_dir())

    # For each image, read and crop the central region, then save to DICOM
    tasks = list(mastersheet.itertuples(index=False, name=None))
    with Pool(processes=12) as pool:
        pool.starmap(_save, tasks)


if __name__ == "__main__":
    main()
