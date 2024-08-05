"""
Model for performing segmentation

"""

import os

from ray import tune
import torch
import numpy as np
import torchio as tio
import torch.utils
from tqdm import tqdm, trange
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


def _pbar(
    data: torch.utils.data.DataLoader, mode: str, notebook: bool
) -> torch.utils.data.DataLoader:
    """
    Get batches wrapped in the right progress bar type based on whether we're in a notebook or not

    """
    assert mode in {"Training", "Validation"}

    progress_bar = tqdm_nb if notebook else tqdm
    return progress_bar(enumerate(data), mode, total=len(data), leave=False)


def _get_data(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the image and labels from each entry in a batch

    """
    # Each batch contains an image, a label and a location (which we don't care about)
    # We also just want to use the data (tio.DATA) from each of these
    x = data["image"][tio.DATA]
    y = data["label"][tio.DATA]

    return x, y


def train_step(
    model: AttentionUnet,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    notebook: bool = False,
) -> tuple[AttentionUnet, list[float]]:
    """
    Train the model for one epoch, on the given batches of data provided as a dataloader

    :param model: the model to train
    :param optimiser: the optimiser to use
    :param loss_fn: the loss function to use
    :param train_data: the training data
    :param device: the device to run the model on
    :param notebook: whether we're running in a notebook or not (to show a progress bar)
    :param return_batch_losses: whether to return the losses for each batch

    :returns: the trained model
    :returns: list of training batch losses

    """
    # If the gradient is too large, we might want to clip it
    # Setting this to some reasonable value might help
    # (find with this, put after loss.backward():)
    # total_norm = 0
    # for p in model.parameters():
    #    if p.grad is not None:
    #        param_norm = p.grad.data.norm(2)
    #        total_norm += param_norm.item() ** 2
    # if total_norm ** 0.5 > max_grad:
    #     print(total_norm ** 0.5)
    max_grad = np.inf

    model.train()

    batch = _pbar(train_data, "Training", notebook)

    train_losses = []
    for _, data in batch:
        x, y = _get_data(data)

        input_, target = x.to(device), y.to(device)

        optimiser.zero_grad()
        out = model(input_)

        loss = loss_fn(out, target)
        train_losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimiser.step()

        batch.set_description(f"Training (loss: {loss.item():.4f})")
    batch.close()

    return model, train_losses


def validation_step(
    model: AttentionUnet,
    loss_fn: torch.nn.Module,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    notebook: bool = False,
) -> tuple[AttentionUnet, list[float]]:
    """
    Find the loss on the validation data

    :param model: the model to train
    :param loss_fn: the loss function to use
    :param train_data: the validation data
    :param device: the device to run the model on
    :param notebook: whether we're running in a notebook or not (to show a progress bar)

    :returns: the trained model
    :returns: validation loss for each batch

    """
    model.eval()

    batch = _pbar(validation_data, "Validation", notebook)

    losses = np.ones(len(validation_data)) * np.nan

    for i, data in batch:
        x, y = _get_data(data)

        input_, target = x.to(device), y.to(device)

        with torch.no_grad():
            out = model(input_)
            loss = loss_fn(out, target)
            losses[i] = loss.item()

        batch.set_description(f"Validation (loss: {loss.item():.4f})")
    batch.close()

    return model, losses


def train(
    model: AttentionUnet,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    checkpoint: bool = False,
    notebook: bool = False,
) -> tuple[AttentionUnet, list[list[float], list[list[float]]]]:
    """
    Train the model for the given number of epochs

    :param model: the model to train
    :param optimiser: the optimiser to use
    :param loss_fn: the loss function to use
    :param train_data: the training data
    :param validation_data: the validation data
    :param device: the device to run the model on
    :param epochs: the number of epochs to train for
    :param lr_scheduler: optional learning rate scheduler to use
    :param checkpoint: whether to checkpoint the model after each epoch
    :param notebook: whether we're running in a notebook or not (to show a progress bar)

    :returns: the trained model
    :returns: list of training batch losses
    :returns: list of validation batch losses

    """
    train_batch_losses = []
    val_batch_losses = []

    for epoch in trange(epochs):
        model, train_batch_loss = train_step(
            model, optimiser, loss_fn, train_data, device=device, notebook=notebook
        )
        train_batch_losses.append(train_batch_loss)

        model, val_batch_loss = validation_step(
            model, loss_fn, validation_data, device=device, notebook=notebook
        )
        val_batch_losses.append(val_batch_loss)

        # Checkpoint the model
        if checkpoint:
            raise NotImplementedError
        #     with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #         path = os.path.join(checkpoint_dir, "checkpoint")
        #         torch.save((model.state_dict(), optimiser.state_dict()), path)

        # We might want to adjust the learning rate during training
        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(np.mean(val_batch_losses[-1]))
            else:
                lr_scheduler.step()

    return model, train_batch_losses, val_batch_losses
