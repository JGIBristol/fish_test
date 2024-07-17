"""
General utility functions, etc.

"""

from .. import util


import math


def model_params():
    """
    Get the model params from the user config file,
    calcuating any extras that might be needed and renaming
    them to be consistent with the monai API

    """
    params = util.userconf()["model_params"]

    # Get the number of channels for each layer by finding the number channels in the first layer
    # and then doing some maths
    start = int(math.sqrt(params["n_initial_filters"]))
    channels_per_layer = [2**n for n in range(start, start + params["n_layers"])]
    params["channels"] = channels_per_layer

    # Convolution stride is always the same, apart from in the first layer where it's implicitly 1
    # (to preserve the size of the input)
    strides = [params["stride"]] * (params["n_layers"] + 1)
    params["strides"] = strides

    # Rename some of the parameters to be consistent with the monai API
    params["out_channels"] = params.pop("n_classes")

    # Remove unused parameters
    params.pop("n_initial_filters")
    params.pop("n_layers")
    params.pop("stride")

    return params
