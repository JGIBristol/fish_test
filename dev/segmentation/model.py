"""
Model for performing segmentation

"""

import torch
import numpy as np
import torch.utils
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from torchviz import make_dot
from monai.networks.nets import AttentionUnet

from . import util
from ..util import userconf


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


def optimiser(model: AttentionUnet) -> torch.optim.Optimizer:
    """
    Get the right optimiser by reading the user config file

    :param model: the model to optimise
    :returns: the optimiser

    """
    user_config = userconf()
    return getattr(torch.optim, user_config["optimiser"])(
        model.parameters(), user_config["learning_rate"]
    )


def _pbar(data: torch.utils.DataLoader, notebook: bool) -> type:
    """
    Get batches wrapped in the right progress bar type based on whether we're in a notebook or not

    """
    progress_bar = tqdm_nb if notebook else tqdm
    return progress_bar(enumerate(data), "Training", total=len(data), leave=False)


def train_step(
    model: AttentionUnet,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    notebook: bool = False,
) -> tuple[AttentionUnet, float]:
    """
    Train the model for one epoch, on the given batches of data provided as a dataloader

    :param model: the model to train
    :param optimiser: the optimiser to use
    :param loss_fn: the loss function to use
    :param train_data: the training data
    :param device: the device to run the model on
    :param notebook: whether we're running in a notebook or not (to show a progress bar)

    :returns: the trained model
    :returns: average training loss over the epoch

    """
    model.train()

    batches = _pbar(train_data, notebook)

    train_losses = np.ones(len(batches)) * np.nan
    for i, batch in batches:
        x, y = batch
        input_, target = x.to(device), y.to(device)

        optimiser.zero_grad()
        out = model(input_)

        loss = loss_fn(out, target)
        train_losses[i] = loss.item()

        loss.backward()
        optimiser.step()

        batches.set_description(f"Training (loss: {loss.item():.4f})")
    batches.close()

    return model, np.mean(train_losses)


def validation_step(
    model: AttentionUnet,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    notebook: bool = False,
) -> tuple[AttentionUnet, float]:
    """
    Run the model on one epoch of validation data to check the loss

    """
    model.eval()

    batches = _pbar(validation_data, notebook)

    losses = np.ones(len(validation_data)) * np.nan

    for i, batch in enumerate(batches):
        x, y = batch
        input_, target = x.to(device), y.to(device)

        with torch.no_grad():
            out = model(input_)
            loss = loss_fn(out, target)
            losses[i] = loss.item()

        batch.set_description(f"Validation (loss: {loss.item():.4f})")
    batches.close()

    return model, np.mean(losses"""  """)
