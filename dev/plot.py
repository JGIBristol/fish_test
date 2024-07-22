"""
Plotting helpers

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _class_cmap(n_classes: int) -> plt.cm.ScalarMappable:
    """
    Colormap for a given number of classes, where the first color is transparent

    """
    viridis = plt.cm.get_cmap("viridis", n_classes)
    colors = viridis(np.linspace(0, 1, n_classes))
    colors[0] = (1, 1, 1, 0)

    return ListedColormap(colors)


def plot_arr(
    arr: np.ndarray, mask: np.ndarray = None
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plot slices of a 3d array

    """
    if mask is not None:
        if mask.shape != arr.shape:
            raise ValueError("Array and mask must have the same shape")

    indices = np.floor(np.arange(0, arr.shape[0], arr.shape[0] // 16)).astype(int)
    vmin, vmax = np.min(arr), np.max(arr)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray", vmin=vmin, vmax=vmax)
        if mask is not None:
            ax.imshow(mask[i], cmap=_class_cmap(len(np.unique(mask))))
        ax.axis("off")
        ax.set_title(i)

    fig.tight_layout()

    return fig, axes
