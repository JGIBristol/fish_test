"""
Read, crop, save, etc. images

"""

import re
import pathlib
import warnings
from functools import cache
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchio as tio

from . import util


@cache
def user_config() -> dict:
    """
    Read the user configuration file

    """
    warnings.warn(
        "This function is deprecated. Use util.user_config instead.", DeprecationWarning
    )
    return util.userconf()


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

    """
    return torch.from_numpy(img).float().unsqueeze(0)


def pytorch2img(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array image

    :param tensor: PyTorch tensor of the image.
    :returns: NumPy array of the image.
    """
    return tensor.squeeze().detach().cpu().numpy()


def random_transforms() -> tio.transforms.Compose:
    """
    The random transforms to apply to the images

    """
    return tio.Compose(
        [
            tio.RandomFlip(axes=(0), flip_probability=0.5),
            tio.RandomAffine(p=1),
            tio.RandomBlur(p=0.4),
            # Removed this one as I'm not sure its relevant
            # tio.RandomBiasField(0.75, order=4, p=0.5),
            tio.RandomNoise(1, 0.02, p=0.5),
            tio.RandomGamma((-0.3, 0.3), p=0.25),
            tio.ZNormalization(masking_method="label", p=1),
            # Removed this as it really messes the image up sometimes
            # tio.OneOf(
            #     {
            #         tio.RescaleIntensity(percentiles=(0, 98)): 0.25,
            #         tio.RescaleIntensity(percentiles=(2, 100)): 0.25,
            #         tio.RescaleIntensity(percentiles=(0.5, 99.5)): 0.25,
            #     }
            # ),
        ]
    )


def subject(image: np.ndarray, mask: np.ndarray) -> tio.Subject:
    """
    Create a TorchIO subject from an image and mask

    :param image: Image to use
    :param mask: Mask to use

    """
    if not image.shape == mask.shape:
        raise ValueError("Image and mask must have the same shape")

    return tio.Subject(
        image=tio.ScalarImage(tensor=img2pytorch(image)),
        label=tio.LabelMap(tensor=img2pytorch(mask)),
    )


def subject_dataset(
    subjects: list[tio.Subject], transform: tio.transforms.augmentation.RandomTransform
) -> tio.SubjectsDataset:
    """
    Create a TorchIO SubjectsDataset from a list of subjects

    :param subjects: List of subjects
    :param transform: Transform to apply to the subjects

    :returns: SubjectsDataset

    """
    return tio.SubjectsDataset(subjects, transform=transform)
