"""
Model for performing segmentation

"""

from monai.networks.nets import AttentionUnet

from . import util


def model() -> AttentionUnet:
    """
    U-Net model for segmentation

    """
    return AttentionUnet(**util.model_params())
