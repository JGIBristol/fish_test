"""
Things for localising the fish jaw

"""

from typing import Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


def _smooth(data, *, smooth_params: dict = None):
    """Smooth some data a bit"""
    if smooth_params is None:
        smooth_params = {}
    if "window_length" not in smooth_params:
        smooth_params["window_length"] = 21
    if "polyorder" not in smooth_params:
        smooth_params["polyorder"] = 3

    # Smooth the profile
    return savgol_filter(data, **smooth_params)


def _slope(data):
    """slope of some data"""
    x = np.arange(len(data))

    def f(x, a, b):
        return a * x + b

    popt, _ = curve_fit(f, x, data)
    return popt[0]


def gradient_peaks(
    n_white_profile: np.ndarray,
    *,
    window_size: int = 25,
    smooth_params: dict = None,
    return_smooth: bool = False,
    return_grad: bool = False,
    return_diff: bool = False,
) -> Union[
    tuple[np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    From a 1d array showing how many white pixels there are per slice in a CT scan,
    find the gradient of sub-windows and find peaks in the first derivative of this.

    This will give us points where the gradient abruptly increases - which we expect to see
    as the jaw is encountered

    :param n_white_profile: The number of white pixels per slice
    :param window_size: The size of the window to use for the gradient calculation
    :param smooth_params: additional parameters for the smoothing function
    :param return_grad: return the gradients as well as the peaks
    :param return_smooth: return the smoothed array as well as the peaks
    :param return_diff: return the diff array as well as the peaks

    :returns: peaks in the derivative of the gradient. These cannot be within the first
    or last sub-section of the array
    :returns: the diff array if return_diff is True
    :returns: the smooth array if return_smooth is True

    """
    smoothed = _smooth(n_white_profile, smooth_params=smooth_params)

    # Find the gradient of sub-windows and pad
    slopes = [
        _slope(smoothed[i : i + window_size])
        for i in range(len(smoothed) - window_size)
    ]
    slopes = slopes + [0] * window_size

    assert len(slopes) == len(
        smoothed
    ), "Slopes and smoothed profile should be the same length"  # Just to check

    # Find the diff and normalise it
    diff = np.concatenate([np.diff(slopes), [0]])
    assert len(diff) == len(slopes), "Diff and slopes should be the same length"

    # Normalise only using those from the central region
    reject_size = 250
    diff = diff / np.max(diff[reject_size:-reject_size])

    # Find peaks in the diff
    peaks = find_peaks(diff, height=0.1)[0]

    # Remove ones we don't want
    peaks = peaks[(peaks > reject_size) & (peaks < len(n_white_profile) - reject_size)]

    # Build the return value
    retval = [peaks]
    if return_smooth:
        retval.append(smoothed)
    if return_grad:
        retval.append(slopes)
    if return_diff:
        retval.append(diff)
    return tuple(retval)


def jaw_peak(peaks, *, n_req: int = 3, width: int = 200) -> int:
    """
    From a list of peaks, find the one that is most likely to be the jaw

    :param peaks: The peaks in the gradient of the number of white pixels per slice
    :param n_req: number of peaks required within the width
    :param width: the width within which we require the peaks

    :returns: the peak that is most likely to be the jaw
    :raises ValueError: if no accepted peaks are found

    """
    for peak in peaks[::-1]:
        # Count the number within width
        n = len([p for p in peaks if (peak - p < width) and (peak != p)])
        if n >= n_req:
            return peak

    raise ValueError("No accepted peaks found")
