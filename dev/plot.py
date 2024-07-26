"""
Plotting helpers

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.networks.nets import AttentionUnet


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

    # If mask only has a few unique values, we want to use a nice qualitative colormap
    # Otherwise just use a sequantial one
    mask_cmap = (
        _class_cmap(len(np.unique(mask)))
        if mask is not None and len(np.unique(mask)) < 10
        else "hot_r"
    )

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray", vmin=vmin, vmax=vmax)
        if mask is not None:
            ax.imshow(mask[i], cmap=mask_cmap, alpha=0.5)
        ax.axis("off")
        ax.set_title(i)

    fig.tight_layout()

    return fig, axes


def plot_nth_filter(
    model: AttentionUnet, n: int
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plot the nth filter of the first convolutional layer of a model

    """
    activations = []

    def activation_hook(module, in_tensor, out_tensor):
        activations.append(out_tensor)

    first_layer = model.model[0]
    first_layer.register_forward_hook(activation_hook)

    # Maybe the tensor size here should match the patch size
    input_tensor = torch.rand(1, 1, 128, 128, 128).to(next(model.parameters()).device)

    with torch.no_grad():
        model.eval()
        model(input_tensor)

    first_layer_activations = activations[0].squeeze()

    return plot_arr(first_layer_activations[n].cpu().numpy() * 255)
