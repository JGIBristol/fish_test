"""
Plotting helpers

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_arr(arr: np.ndarray) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plot slices of a 3d array

    """
    indices = np.floor(np.arange(0, arr.shape[0], arr.shape[0] // 16)).astype(int)
    vmin, vmax = np.min(arr), np.max(arr)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(i)

    fig.tight_layout()

    return fig, axes
