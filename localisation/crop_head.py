"""
Crop out the fish head by looking at the greyscale intensity of the image and making some assumptions about the fish's orientation

"""

import os
import sys
import pathlib
import argparse
from multiprocessing import Pool, Manager

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dev import image_io, util, thresholding, plot as plot_lib


def n_white_per_slice(img_arr: np.ndarray, threshold_pct: float) -> np.ndarray:
    """
    Find how many white pixels there are per slice with the given threshold

    :param img_arr: The image array
    :param threshold_pct: The threshold, as a quantile in percent

    """
    # Find the threshold
    threshold = np.quantile(img_arr, 1 - (threshold_pct / 100))

    # Count the number of white pixels per slice
    return np.sum(img_arr > threshold, axis=(1, 2))


def _n_white_per_slice(
    img_arr: np.ndarray, threshold: float, shared_list: list
) -> None:
    shared_list.append(n_white_per_slice(img_arr, threshold))


def avg_profile(
    img_arr: np.ndarray, thresholds: list[float], n_jobs: int
) -> np.ndarray:
    """
    For many thresholds, find the number of white pixels per slice
    and take the average of them

    """
    shared_list = Manager().list()

    with Pool(n_jobs) as pool:
        pool.starmap(
            _n_white_per_slice,
            [(img_arr, threshold, shared_list) for threshold in thresholds],
        )

    return np.mean(shared_list, axis=0)


def _plot_profile(profile: np.ndarray, plot_dir: str) -> None:
    """
    Plot the intensity profile

    """
    fig, axis = plt.subplots()

    axis.plot(profile)
    fig.suptitle("Number of white pixels per slice, averaged over thresholds")

    axis.set_xlabel("Slice No")
    axis.set_ylabel("Avg number of white pixels")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}profile.png")


def main(*, img_n: int, n_jobs: int, plot: bool):
    """
    Read in the image, count how many white pixels there are for various thresholds and take the average

    Then find peaks in the allowed region (not near the edges, profile decreasing, not too small) and
    choose one of these peaks

    Then from the corresponding image slice, find the x/y window that contains the head and crop it out

    """
    if plot:
        plot_dir = f"plots/{img_n}/"
        if not os.path.exists("plots"):
            os.mkdir("plots")
        os.mkdir(plot_dir)

    # Read in the image
    img_arr = image_io.read_tiffstack(
        img_n, n_jobs=None
    )  # Don't use threads here because it doesn't help
    if plot:
        fig, _ = plot_lib.plot_stack(img_arr)
        fig.savefig(f"{plot_dir}raw_stack.png")

    # Equalise the image, setting the number of pixels to be saturated
    img_arr = thresholding.equalise(img_arr, saturated_pct=0.35)
    if plot:
        fig, _ = plot_lib.plot_stack(img_arr)
        fig.savefig(f"{plot_dir}eq_stack.png")

    # For various thresholds, count the number of white pixels
    thresholds = [0.50, 0.65, 0.80, 0.95, 1.1, 1.25, 1.4]
    profile = avg_profile(img_arr, thresholds, n_jobs)
    if plot:
        _plot_profile(profile, plot_dir)

    # Find peaks
    # Define our allowed regions and find where peaks are allowed
    # Choose a peak and find the corresponding slice
    # Choose the x/y window


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "img_n",
        type=int,
        help="The image number to crop the head from",
    )
    parser.add_argument(
        "--n_jobs",
        "-n",
        type=int,
        default=6,
        help="Number of threads to use for thresholding.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the images at each stage of the process",
        default=False,
    )

    main(**vars(parser.parse_args()))
