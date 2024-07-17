"""
Model for performing segmentation

"""

import torch
from torchviz import make_dot
from monai.networks.nets import AttentionUnet

from . import util


def model() -> AttentionUnet:
    """
    U-Net model for segmentation

    """
    return AttentionUnet(**util.model_params())


def draw_model(model: AttentionUnet, path: str) -> None:
    """
    Create an image of the architecture of the model at the given path
    Also creates a dot file

    :param: the model
    :param: the path to save the image, e.g. "model.png"

    """
    path, fmt = path.rsplit(".", 1)

    # Add dummy input to the model to generate the image
    params = util.util.userconf()["model_params"]
    dummy_input = torch.randn(
        1, params["in_channels"], *[128 for _ in range(params["spatial_dims"])]
    )

    make_dot(model(dummy_input), params=dict(model.named_parameters())).render(
        path, format=fmt
    )
