"""
Threshold out useful parts of the image stack (probably the bones)

"""

import logging

import numpy as np
from scipy.signal import find_peaks

from skimage.exposure import rescale_intensity


def equalise(stack: np.ndarray, *, saturated_pct: 0.5) -> np.ndarray:
    """
    Equalise the stack of images

    :param stack: The stack of images

    """
    v_min, v_max = np.percentile(stack.flat, (saturated_pct, 100 - saturated_pct))

    return rescale_intensity(stack, in_range=(v_min, v_max))


def img_thresholds(
    stack: np.ndarray, thresholds: list[float], *, return_nwhite: bool = False
) -> list[int] | tuple[list[int], list[int]]:
    """
    Find the significant thresholds for an image

    Using the provided thresholds, count how many pixels are above each threshold.
    Then use the change in these to determine the significant thresholds.

    :param stack: The stack of images
    :param thresholds: The thresholds to use

    """
    n_white = [np.sum(stack > threshold) for threshold in thresholds]

    log_diff = np.log(-np.diff(n_white))

    peaks = thresholds[np.array(find_peaks(log_diff)[0])]

    if len(peaks) != 4:
        # We expect four peaks as the binding, tube, flesh and bone are removed
        logging.warning(f"Expected 4 peaks, got {len(peaks)}")

    if return_nwhite:
        return peaks, n_white
    return peaks
