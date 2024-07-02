"""
Crop out the fish head by looking at the greyscale intensity of the image and making some assumptions about the fish's orientation

"""

import os
import sys
import pathlib
import argparse
from PIL import Image
from multiprocessing import Pool, Manager

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as sp_find_peaks, convolve

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dev import image_io, thresholding, plot as plot_lib


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


def _plot_profile(profile: np.ndarray, plot_dir: pathlib.Path) -> None:
    """
    Plot the intensity profile

    """
    fig, axis = plt.subplots()

    axis.plot(profile)
    fig.suptitle("Number of white pixels per slice, averaged over thresholds")

    axis.set_xlabel("Slice No")
    axis.set_ylabel("Avg number of white pixels")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/profile.png")


def find_peaks(profile: np.ndarray) -> np.ndarray:
    """
    Find peaks in the profile

    """
    # Smooth a little
    tophat_width = 10
    smoothed = np.convolve(profile, np.ones(tophat_width) / tophat_width, mode="same")

    peaks = sp_find_peaks(smoothed, height=0.2, prominence=0.1, distance=50)[0]
    return peaks


def _plot_peaks(
    peaks: np.ndarray, profile: np.ndarray, plot_dir: pathlib.Path, filename: str
) -> None:
    """
    Plot the peaks on the profile

    """
    fig, axis = plt.subplots()

    axis.plot(profile)
    for peak in peaks:
        axis.axvline(peak, color="red")

    fig.suptitle("Peaks in the profile")
    axis.set_xlabel("Slice No")
    axis.set_ylabel("Avg number of white pixels")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/{filename}.png")


def _smoothed(profile: np.ndarray) -> np.ndarray:
    n = 500
    tophat = np.ones(n) / n
    return np.convolve(profile, tophat, mode="same")


def _decreasing(profile: np.ndarray, *, plot_dir: pathlib.Path) -> np.ndarray:
    """
    Smooth and find where the profile is decreasing
    Plots if plot_dir is provided

    """
    smoothed = _smoothed(profile)

    if plot_dir:
        fig, axis = plt.subplots()
        axis.plot(smoothed)

        fig.suptitle("Smoothed profile")
        axis.set_xlabel("Slice No")
        axis.set_ylabel("Avg number of white pixels")

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/smoothed.png")

    return np.concatenate([[False], np.diff(smoothed) < 0])


def _allowed_regions(profile: np.ndarray, *, plot_dir: pathlib.Path) -> np.ndarray:
    """
    Define the allowed regions for peaks in the profile

    Plots if plot_dir is provided

    """
    # Smooth the profile amd allow only points where the gradient is decreasing
    allowed = _decreasing(profile, plot_dir=plot_dir)

    # and the profile is not too small
    allowed[profile < np.max(profile) / 10] = False

    # and the peak is not near the edges
    allowed[:250] = False
    allowed[-250:] = False

    # Plot the allowed regions
    if plot_dir:
        fig, axis = plt.subplots()
        axis.plot(profile)
        axis.plot(_smoothed(profile), "k", alpha=0.2)
        axis.plot(np.arange(len(profile))[allowed], profile[allowed], "k.")

        fig.suptitle("Allowed Regions for peaks")
        axis.set_xlabel("Slice No")
        axis.set_ylabel("Avg number of white pixels")

        fig.savefig(f"{plot_dir}/allowed.png")

    return allowed


def _crop(img_2d: np.ndarray, co_ords: tuple[int, int], window_size: tuple[int, int]):
    """
    Crop an image

    """
    return img_2d[
        co_ords[0] : min(co_ords[0] + window_size[0], img_2d.shape[0]),
        co_ords[1] : min(co_ords[1] + window_size[1], img_2d.shape[1]),
    ]


def find_window(
    img: np.ndarray, window_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """ """
    kernel = np.ones(window_size)

    # Count the number of 1s in each sub-window
    conv_result = convolve(img, kernel, mode="valid")

    # Find the index of the maximum value in the convolution result
    max_index = np.unravel_index(np.argmax(conv_result), conv_result.shape)

    # Extract the sub-window using the index
    # Note: The end index is start index + size of the sub-window in each dimension
    sub_window = _crop(img, max_index, window_size)

    return sub_window, max_index


def main(*, img_n: int, n_jobs: int, plot: bool):
    """
    Read in the image, count how many white pixels there are for various thresholds and take the average

    Then find peaks in the allowed region (not near the edges, profile decreasing, not too small) and
    choose one of these peaks

    Then from the corresponding image slice, find the x/y window that contains the head and crop it out

    """
    out_dir = pathlib.Path(f"cropped/{img_n}/").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        plot_dir = out_dir / "plots/"
        if not plot_dir.is_dir():
            plot_dir.mkdir(exist_ok=True)

    # Read in the image
    img_arr = image_io.read_tiffstack(
        img_n, n_jobs=None
    )  # Don't use threads here because it doesn't help
    if plot:
        print("Plotting images")
        fig, _ = plot_lib.plot_arr(img_arr)
        fig.savefig(f"{plot_dir}/raw_stack.png")

    # Equalise the image, setting the number of pixels to be saturated
    img_arr = thresholding.equalise(img_arr, saturated_pct=0.35)
    if plot:
        print("Plotting images")
        fig, _ = plot_lib.plot_arr(img_arr)
        fig.savefig(f"{plot_dir}/eq_stack.png")

    # For various thresholds, count the number of white pixels
    thresholds = [0.50, 0.65, 0.80, 0.95, 1.1, 1.25, 1.4]
    profile = avg_profile(img_arr, thresholds, n_jobs)
    if plot:
        print("Plotting profile")
        _plot_profile(profile, plot_dir)

    # Find peaks
    peaks = find_peaks(profile)
    if plot:
        print("Plotting all peaks")
        _plot_peaks(peaks, profile, plot_dir, "peaks")

    # Define our allowed regions and find where peaks are allowed
    allowed = _allowed_regions(profile, plot_dir=plot_dir if plot else "")

    # Choose a peak and find the corresponding slice
    allowed_peaks = [peak for peak in peaks if allowed[peak]]
    if plot:
        print("Plotting allowed peaks")
        _plot_peaks(allowed_peaks, profile, plot_dir, "allowed_peaks")
    peak = allowed_peaks[-1]

    # Choose the x/y window
    window_size = (250, 250)
    sub_window, crop_coords = find_window(img_arr[peak], window_size)
    if plot:
        fig, axis = plt.subplots()
        axis.imshow(sub_window, cmap="gray")
        fig.savefig(f"{plot_dir}/sub_window.png")

    # Convert to uint8
    img_arr = (img_arr * 255).astype(np.uint8)

    # Save the sub-window as images
    n_z = 250
    for z in range(peak - n_z // 2, peak + n_z // 2):
        Image.fromarray(_crop(img_arr[z], crop_coords, window_size)).save(
            f"{out_dir}/sub_window_{peak + z}.jpg"
        )  # Filename corresponds to the slice number


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
