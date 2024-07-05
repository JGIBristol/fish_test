"""
Crop out the fish head by looking at the greyscale intensity of the image and making some assumptions about the fish's orientation

"""

import os
import sys
import time
import pathlib
import argparse
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dev import image_io, thresholding, plot as plot_lib, localisation


def n_white_per_slice(
    img_arr: np.ndarray, threshold_pct: float, plot_dir: pathlib.Path
) -> np.ndarray:
    """
    Find how many white pixels there are per slice with the given threshold

    :param img_arr: The image array
    :param threshold_pct: The threshold, as a quantile in percent

    """
    # Find the threshold
    threshold = np.quantile(img_arr, 1 - (threshold_pct / 100))

    # Count the number of white pixels per slice
    thresholded = img_arr > threshold
    if plot_dir is not None:
        fig, _ = plot_lib.plot_arr(thresholded)
        fig.savefig(f"{plot_dir}/thresholded.png")

    return np.sum(thresholded, axis=(1, 2))


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


def _plot_peaks(
    peaks: np.ndarray,
    profile: np.ndarray,
    plot_dir: pathlib.Path,
    filename: str,
    grad: np.ndarray,
    diff: np.ndarray,
) -> None:
    """
    Plot the peaks on the profile

    """
    fig, axes = plt.subplots(3, 1)

    axes[0].plot(profile / np.max(profile))
    axes[1].plot(grad / np.max(grad), "r")
    axes[2].plot(diff / np.max(diff), "k")

    axes[0].set_ylabel("N white pxl")
    axes[1].set_ylabel("Gradient")
    axes[2].set_ylabel("Diff")

    for axis in axes:
        for peak in peaks:
            axis.axvline(peak, color="red")
        axis.set_xlabel("Slice No")

    fig.suptitle("Peaks in the profile")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/{filename}.png")


def _crop(img_2d: np.ndarray, co_ords: tuple[int, int], window_size: tuple[int, int]):
    """
    Crop an image

    """
    half_width = window_size[0] // 2
    half_height = window_size[1] // 2
    return img_2d[
        max(int(co_ords[1] - half_height), 0) : min(
            int(co_ords[1] + half_height), img_2d.shape[0]
        ),
        max(int(co_ords[0] - half_width), 0) : min(
            int(co_ords[0] + half_width), img_2d.shape[1]
        ),
    ]


def find_window(
    img: np.ndarray, window_size: tuple[int, int], threshold: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Find the window in the (2d) image with the most white pixels when it is thresholded

    """
    kernel = np.ones(window_size)

    # Count the number of 1s in each sub-window
    conv_result = convolve(img > threshold, kernel, mode="same")

    # Find all indices where the convolution result matches the maximum value
    max_value = np.max(conv_result)
    max_indices = np.argwhere(conv_result == max_value)

    # Find the average of these - to hopefully get the middle of the jaw
    max_index = np.mean(max_indices, axis=0).astype(int)

    sub_window = _crop(img, max_index, window_size)

    return sub_window, max_index, conv_result


def main(*, img_n: int, plot: bool):
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
    print(f"Reading {img_n}")
    img_arr = image_io.read_tiffstack(
        img_n, n_jobs=None
    )  # Don't use threads here because it doesn't help
    if plot:
        print("Plotting images")
        fig, _ = plot_lib.plot_arr(img_arr)
        fig.savefig(f"{plot_dir}/raw_stack.png")

    # Equalise the image, setting the number of pixels to be saturated
    start_t = time.perf_counter()
    img_arr = thresholding.equalise(img_arr, saturated_pct=0.35)
    if plot:
        print("Plotting images")
        fig, _ = plot_lib.plot_arr(img_arr)
        fig.savefig(f"{plot_dir}/eq_stack.png")
        plt.close(fig)

    # For various thresholds, count the number of white pixels
    threshold = 0.50
    profile = n_white_per_slice(img_arr, threshold, plot_dir=plot_dir if plot else "")
    if plot:
        print("Plotting profile")
        _plot_profile(profile, plot_dir)

    # Find peaks
    peaks, smoothed, grad, diff = localisation.gradient_peaks(
        profile, return_smooth=True, return_grad=True, return_diff=True
    )
    if plot:
        print("Plotting all peaks")
        _plot_peaks(peaks, smoothed, plot_dir, "peaks", grad, diff)

    # Choose a peak and find the corresponding slice
    jaw_peak = localisation.jaw_peak(peaks)
    if plot:
        print("Plotting jaw peak")
        _plot_peaks([jaw_peak], profile, plot_dir, "jaw_peak", grad, diff)

    # Choose the x/y window
    window_size = (250, 250)
    sub_window, crop_coords, conv = find_window(
        img_arr[jaw_peak], window_size, threshold
    )
    print(f"{crop_coords=}")
    if plot:
        # Plot the sub window
        fig, axis = plt.subplots()
        axis.imshow(sub_window, cmap="grey")
        fig.savefig(f"{plot_dir}/sub_window.png")
        plt.close(fig)

        # Plot the convolution
        fig, axis = plt.subplots()
        axis.imshow(conv, cmap="gist_grey")
        axis.plot(crop_coords[1], crop_coords[0], "ro", markersize=10)
        fig.savefig(f"{plot_dir}/convolution.png")

        print(f"{img_arr.shape=}")
        print(f"{conv.shape=}")
        fig.axis = plt.subplots()
        axis.imshow(img_arr[jaw_peak], cmap="gray", alpha=0.6)
        mappable = axis.imshow(conv, cmap="Wistia", alpha=0.5)
        fig.colorbar(mappable)
        fig.savefig(f"{plot_dir}/heatmap.png")
        plt.close(fig)

    # Convert to uint8
    img_arr = (img_arr * 255).astype(np.uint8)

    # Save the sub-window as images
    n_z = 250
    for z in range(jaw_peak - n_z // 2, jaw_peak + n_z // 2):
        Image.fromarray(_crop(img_arr[z], crop_coords, window_size)).save(
            f"{out_dir}/sub_window_{z}.jpg"
        )  # Filename corresponds to the slice number

    print(
        f"Total after reading{' (including plotting) ' if plot else ''}: {time.perf_counter() - start_t:.2f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "img_n",
        type=int,
        help="The image number to crop the head from",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the images at each stage of the process",
        default=False,
    )

    main(**vars(parser.parse_args()))
