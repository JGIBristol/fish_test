"""
Read, crop, save, etc. images

"""

import re
import yaml
import pathlib
from functools import cache
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch


@cache
def user_config() -> dict:
    """
    Read the user configuration file

    Should be in util/

    """
    with open(str(pathlib.Path(__file__).parents[1] / "userconf.yml")) as file:
        return yaml.safe_load(file)


@cache
def ct_scan_dir() -> pathlib.Path:
    """
    Path to the directory

    """
    return (
        pathlib.Path(user_config()["rdsf_dir"])
        / "DATABASE"
        / "uCT"
        / "Wahab_clean_dataset"
        / "low_res_clean_v3"
    )


def img_dir(img_n: int) -> pathlib.Path:
    """
    Path to the directory of the image

    :param img_n: "old_n" in the mastersheet

    """
    return ct_scan_dir() / f"{img_n:03}" / "reconstructed_tifs"


def parse_roi(roi_str: str) -> tuple[int, int, int]:
    """
    Parse the ROI string [Z Y X] into a tuple of ints

    :param roi_str: ROI string from the mastersheet

    """
    return tuple(map(int, re.findall(r"\d+", roi_str)))


def _init_tiffstack_arr(img_paths: list[pathlib.Path]) -> np.ndarray:
    """
    Initialize the array to hold the stack of tiff images, and populate the first image

    """
    img0 = cv2.imread(str(img_paths[0]), cv2.IMREAD_GRAYSCALE)
    shape = img0.shape

    retval = np.empty((len(img_paths), *shape))
    retval[0] = img0

    return retval


def _read_tiffstack_singlethread(img_paths: list[pathlib.Path]) -> np.ndarray:
    """
    Single threaded version of read_tiffstack

    """

    # Read the first image to determine the shape
    retval = _init_tiffstack_arr(img_paths)

    # The first image is already populated, so just do the rest in order
    for i, img_path in enumerate(img_paths[1:], start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        retval[i] = img

    return retval


def _read_tiffstack_multithread(
    img_paths: list[pathlib.Path], n_jobs: int
) -> np.ndarray:
    """
    Multiple threads

    """
    retval = _init_tiffstack_arr(img_paths)

    def _write_img2array(i: int, path: pathlib.Path, arr: np.ndarray) -> None:
        """Write the image to path"""
        arr[i] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(_write_img2array, i, path, retval)
            for i, path in enumerate(img_paths[1:], start=1)
        ]

        for future in futures:
            future.result()

    return retval


def read_tiffstack(n: int, *, n_jobs: int = None) -> np.ndarray:
    """
    Read a stack of .tiff images from the right directory; assumed the zebrafish osteoarthritis RDSF is mounted at the provided dir

    :param n: scan number; old_n in metadata

    """
    img_paths = sorted(img_dir(n).glob("*.tiff"))

    if n_jobs is None:
        return _read_tiffstack_singlethread(img_paths)
    else:
        return _read_tiffstack_multithread(img_paths, n_jobs)


def img2pytorch(img: np.ndarray) -> torch.tensor:
    """
    Convert an image to a PyTorch tensor

    Assumes the image has values between 0 and 255

    """
    return torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
