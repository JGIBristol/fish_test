"""
Crop arrays according to the centres in the mastersheet

"""

import os
import tqdm
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
    print(img_dir)

    # Parse the coordinates
    z, x, y = image_io.parse_roi(region)

    # Choose which images to read
    start_i = z - window_size // 2

    # Read these images and copy the right pixels to the out array
    for i in tqdm.tqdm(range(window_size)):
        img_path = f"{img_dir}/{img_n:03}_{start_i + i:04}.tiff"
        assert os.path.exists(img_path), img_path

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        cropped[i] = img[
            x - window_size // 2 : x + window_size // 2,
            y - window_size // 2 : y + window_size // 2,
        ]

    return cropped


def main():
    """
    For each stack of tifs, read the mastersheet to find the jaw centre, read the right images and crop them
    Then save them as a DICOM

    """
    # Read the relevant columns in the mastersheet
    mastersheet = metadata.mastersheet()[["old_n", "jaw_center"]].set_index("old_n")

    # Choose a directory for images to be saved
    out_dir = "cropped_dicoms/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # For each image, read and crop the central region, then save to DICOM
    for img_n, jaw_center in mastersheet.itertuples():
        img = _cropped_img(img_n, jaw_center)
        np.save(f"{out_dir}{img_n:03}.npy", img)


if __name__ == "__main__":
    main()
