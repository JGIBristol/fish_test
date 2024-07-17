"""
Model for performing segmentation

"""

import monai
import monai.networks
from monai.networks.nets import AttentionUnet


# Attention U-Net from monai
def model(params: dict) -> AttentionUnet:
    """
    U-Net model for segmentation

    """
    return AttentionUnet(spatial_dims=3, in_channels=1, out_channels=
