"""
Try to make a model train on a very simple example

"""

import os
from typing import Iterator, Sequence, Literal

import torch
import tifffile
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from monai.losses import TverskyLoss
from monai.networks.nets import AttentionUnet
from matplotlib.colors import LinearSegmentedColormap

DEVICE = "cuda"
RDSF_DIR = "/home/mh19137/zebrafish_rdsf/"
ROI_SIZE = 192, 192, 192


def crop(image: np.ndarray, roi_centre: tuple[int, int, int]) -> np.ndarray:
    """
    Crop an image around a given centre

    :param image: Image to crop
    :param roi_centre: Centre of the ROI, (z, x, y)

    :returns: Cropped image, as a slice

    """
    d, w, h = ROI_SIZE
    z, y, x = roi_centre

    return image[
        z - d // 2 : z + d // 2, x - h // 2 : x + h // 2, y - w // 2 : y + w // 2
    ]


def read_image(n: int, centre: tuple[int, int, int]) -> np.ndarray:
    """
    Read in one of the scans to numpy

    """
    dump_dir = "scan_dumps/"
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)
    output_path = os.path.join(dump_dir, f"{n:03}.tif")

    # If the file exists, read it in
    if os.path.exists(output_path):
        return tifffile.imread(f"scan_dumps/{n:03}.tif")

    # Otherwise, slowly read it from disk, crop it and then dump it to the right path
    input_path = os.path.join(
        RDSF_DIR, r"1Felix and Rich make models", "wahabs_scans", f"{n:03}.tif"
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find file {input_path}")

    scan = tifffile.imread(input_path)

    cropped = crop(scan, centre)
    with open(output_path, "wb") as f:
        tifffile.imwrite(f, cropped)

    return cropped


def read_mask(n: int, centre: tuple[int, int, int]) -> np.ndarray:
    """
    Read in a mask to numpy

    """
    dump_dir = "mask_dumps/"
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)
    output_path = os.path.join(dump_dir, f"{n:03}.tif")

    if os.path.exists(output_path):
        return tifffile.imread(output_path)

    input_path = os.path.join(
        RDSF_DIR,
        "1Felix and Rich make models",
        "Training dataset Tiffs",
        f"{n:03}_0000.labels.tif",
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find file {input_path}")

    full_mask = tifffile.imread(input_path)
    cropped = crop(full_mask, centre)

    with open(output_path, "wb") as f:
        tifffile.imwrite(f, cropped)

    return cropped


def transforms() -> tio.Transform:
    """
    Define the transforms I'm using

    """
    return tio.Compose(
        [
            tio.RandomFlip(axes=(0), flip_probability=0.5),
            tio.RandomAffine(p=1, degrees=25, scales=0.5),
            # tio.RandomBlur(p=0.4),
            # tio.RandomBiasField(0.75, order=4, p=0.5),
            # tio.RandomNoise(1, 0.02, p=0.5),
            # tio.RandomGamma((-0.3, 0.3), p=0.25),
            # tio.ZNormalization(masking_method="label", p=1),
            # tio.OneOf(
            #     {
            #         tio.RescaleIntensity(percentiles=(0, 98)): 0.25,
            #         tio.RescaleIntensity(percentiles=(2, 100)): 0.25,
            #         tio.RescaleIntensity(percentiles=(0.5, 99.5)): 0.25,
            #     }
            # ),
        ]
    )


def dataloader(
    imgs: list[np.ndarray], masks: list[np.ndarray]
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader of our data

    """
    subjects = []
    for img, mask in zip(imgs, masks):
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        subject = tio.Subject(
            image=tio.Image(
                tensor=img_tensor,
                type=tio.INTENSITY,
            ),
            label=tio.Image(
                tensor=mask_tensor,
                type=tio.LABEL,
            ),
        )
        subjects.append(subject)

    subjectsdataset = tio.SubjectsDataset(subjects, transform=transforms())

    # Not sure why it doesn't work if i set pin_memory=True
    return torch.utils.data.DataLoader(subjectsdataset, batch_size=3, shuffle=True)


def train_step(
    model: AttentionUnet,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data: tio.data.subject.Subject,
) -> tuple[AttentionUnet, float]:
    """
    Perform one training step and return the model and loss function

    """
    img = data[tio.IMAGE][tio.DATA]
    mask = data[tio.LABEL][tio.DATA]
    model.train()

    x = img.to(DEVICE)
    y = mask.to(DEVICE)

    optimiser.zero_grad()
    out = model(x)

    loss = loss_fn(out, y)
    loss.backward()
    optimiser.step()

    return model, loss.item()


def val_step(
    model: AttentionUnet,
    loss_fn: torch.nn.Module,
    data: tio.data.subject.Subject,
) -> tuple[AttentionUnet, float]:
    """
    Perform one validation step and return the model and loss function

    """
    model.eval()

    img = data[tio.IMAGE][tio.DATA].unsqueeze(0)
    mask = data[tio.LABEL][tio.DATA].unsqueeze(0)

    x = img.to(DEVICE)
    y = mask.to(DEVICE)

    with torch.no_grad():
        out = model(x)
        loss = loss_fn(out, y)

    return model, loss.item()


def plot_slices(arr: np.ndarray, mask: np.ndarray) -> plt.Figure:
    """
    Plot slices of a 3d array

    """
    indices = np.floor(np.arange(0, arr.shape[0], arr.shape[0] // 16)).astype(int)
    vmin, vmax = np.min(arr), np.max(arr)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in zip(indices, axes.flat):
        ax.imshow(arr[i], cmap="gray", vmin=vmin, vmax=vmax)
        if mask is not None:
            # Create the colormap
            cdict: dict[
                Literal["red", "green", "blue", "alpha"],
                Sequence[tuple[float, ...]],
            ] = {
                "red": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                "green": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                "blue": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                "alpha": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            }
            cmap = LinearSegmentedColormap("TransparentRed", cdict)

            ax.imshow(mask[i], cmap=cmap, alpha=0.3)

        ax.axis("off")
        ax.set_title(i)

    fig.tight_layout()

    return fig


def axis_layout(n) -> tuple[int, int]:
    """
    Find the most square axis layout

    """
    best_pair = (1, n)
    min_diff = n - 1  # Maximum possible difference initially

    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            pair = (i, n // i)
            diff = abs(pair[0] - pair[1])
            if diff < min_diff:
                min_diff = diff
                best_pair = pair
    return best_pair


def plot_filters(model: AttentionUnet) -> Iterator[tuple[int, int, plt.Figure]]:
    """
    Plot the filters of the first convolutional layer of a model
    """
    first_conv_block = model.model[0]
    first_conv_layer = first_conv_block.conv[0].conv

    if not isinstance(first_conv_layer, torch.nn.Conv3d):
        raise TypeError(
            f"First layer is not a 3d convolutional layer, but {type(first_conv_layer)}"
        )

    filters = first_conv_layer.weight.detach().cpu().numpy()
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    num_filters = filters.shape[0]
    num_channels = filters.shape[1]

    num_channels, num_filters, depth, _, _ = filters.shape
    for i in range(num_channels):
        for j in range(num_filters):
            fig, axes = plt.subplots(*axis_layout(depth), figsize=(12, 12))
            for k, ax in enumerate(axes.flat):
                ax.imshow(filters[i, j, k], cmap="gray")
                ax.axis("off")
                ax.set_title(k)
            yield (i, j, fig)


def plot_loss(train_losses: list[float], val_losses: list[float]) -> None:
    """
    Plot the losses for the training and validation sets

    """
    fig, axis = plt.subplots()
    axis.plot(train_losses, label="Train")
    axis.plot(val_losses, label="Validation")
    axis.legend()

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    fig.tight_layout()

    print("plotting loss")
    fig.savefig("tmp_losscurve.png")
    plt.close(fig)


def dice_coefficient(prediction: np.ndarray, truth: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between the prediction and truth arrays.
    """
    intersection = np.sum(prediction * truth)
    return (2.0 * intersection) / (np.sum(prediction) + np.sum(truth))


def _plot_dice(prediction: np.ndarray, truth: np.ndarray, axis: plt.Axes) -> float:
    """
    Plot Dice accuracy axis and some histograms
    Returns the best threshold

    """
    # Using different values of the threshold, calcluate the Dice accuracy
    thresholds = np.linspace(prediction.min(), prediction.max(), 100)
    dice_scores = []
    for threshold in thresholds:
        prediction_binary = (prediction > threshold).astype(int)
        dice_scores.append(dice_coefficient(prediction_binary, truth))

    # Find the threshold at maximum dice score
    best_threshold = thresholds[np.argmax(dice_scores)]

    # Plot it on the axis
    axis.plot(thresholds, dice_scores)
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Dice Accuracy")
    axis.set_title("Dice accuracy vs threshold")

    return best_threshold


def plot_dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """
    Plot histograms and ROC curve

    Returns the best threshold

    """
    fig = plt.figure(layout="constrained", figsize=(12, 12))
    axes = fig.subplot_mosaic(
        """
        AAAAAA
        AAAAAA
        AAAAAA
        BBBCCC
        """,
    )
    # Plot ROC curve on the big axis
    best_threshhold = _plot_dice(prediction, truth, axes["A"])

    # Plot histograms on the small axes
    _, bins, _ = axes["B"].hist(prediction.flat, bins=200)
    axes["B"].set_title("Class predictions")

    axes["C"].hist(prediction.flat, bins=list(bins))
    axes["C"].set_yscale("log")
    axes["C"].set_title("Class predictions (log scale)")

    # Plot the best threshholds
    for axis in axes.values():
        axis.axvline(best_threshhold, color="red")

    fig.suptitle("Validation Data")

    print("Plotting dice")
    fig.savefig("tmp_dice.png")
    plt.close(fig)

    return best_threshhold


def plot_roc(prediction: np.ndarray, truth: np.ndarray) -> None:
    """
    Plot a ROC curve

    """
    fpr, tpr, _ = roc_curve(truth.flat, prediction.flat)

    # Plot ROC curve
    fig, axis = plt.subplots()
    axis.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {auc(fpr, tpr):.2f})",
    )
    axis.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Receiver Operating Characteristic")
    axis.legend(loc="lower right")

    fig.tight_layout()

    fig.savefig("tmp_roc.png")


def main():
    """
    Get an image and labels
    define a model architecture
    train on this image
    show training loss
    plot ROC curve
    show prediction

    """
    # Doing it like this to make the lines easier to comment out
    train_ns, train_centres = zip(
        *{
            69: [1435, 161, 390],
            70: [1350, 409, 243],
            # 89: [1262, 333, 243], This one is a bit weird
            90: [1636, 301, 403],
            93: [1485, 426, 159],
            # 97: [1436, 269, 172], This one also weird
            488: [1497, 388, 358],
        }.items()
    )

    # Read in the images
    imgs = [
        read_image(n, centre)
        for n, centre in tqdm(zip(train_ns, train_centres), total=len(train_ns))
    ]
    masks = [
        read_mask(n, centre)
        for n, centre in tqdm(zip(train_ns, train_centres), total=len(train_ns))
    ]

    # Some of the masks (e.g. 90) also has other stuff labelled with 2; this is background
    masks = [(mask == 1).astype(int) for mask in masks]

    # Plot the testing images
    img_dir = "tmp_imgs"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img, mask, n in zip(imgs, masks, train_ns):
        plot_slices(img, mask).savefig(f"{img_dir}/test_{n}.png")

    # Choose an image for validation
    val_n = 489
    val_centre = (1400, 246, 271)
    val_img = read_image(val_n, val_centre)
    val_mask = read_mask(val_n, val_centre) == 1

    plot_slices(val_img, val_mask).savefig(f"{img_dir}/val_{val_n}.png")

    val_subject = tio.Subject(
        image=tio.Image(
            tensor=torch.tensor(val_img, dtype=torch.float32).unsqueeze(0),
            type=tio.INTENSITY,
        ),
        label=tio.Image(
            tensor=torch.tensor(val_mask, dtype=torch.float32).unsqueeze(0),
            type=tio.LABEL,
        ),
    )

    # Create a dataloader - this will also apply random transformations to the data
    loader = dataloader(imgs, masks)

    # Define the model
    # For a stride of 2, the spatial dimensions will be halved in each layer
    # this means that with an input size of 160x160x160, we can only have 5 layers
    # as any more than this will have non-integer dimensions which isn't allowed
    # You could remedy this by either padding the input image or cropping to a
    # more sensible size (256x256x256?), but I'm not doing that here
    n_layers = 6
    channels = [2 ** (i + 3) for i in range(n_layers)]
    strides = [2] * (n_layers - 1)
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        dropout=0.1,
        kernel_size=3,
    )
    loss = TverskyLoss(sigmoid=True, alpha=0.5)
    optimiser = torch.optim.NAdam(model.parameters(), lr=2e-3)

    model.to(DEVICE)

    # Train on this image
    n_epochs = 500
    train_losses = []
    val_losses = []
    for i in tqdm(range(n_epochs)):
        batch_train_loss = []
        batch_val_loss = []
        for batch in loader:
            model, train_loss = train_step(model, optimiser, loss, batch)
            batch_train_loss.append(train_loss)

            model, val_loss = val_step(model, loss, val_subject)
            batch_val_loss.append(val_loss)

        train_losses.append(np.mean(batch_train_loss))
        val_losses.append(np.mean(batch_val_loss))

    # Show training loss
    plot_loss(train_losses, val_losses)

    # Show prediction on the next item from the training set
    batch = next(iter(loader))
    img = (
        batch[tio.IMAGE][tio.DATA][0].unsqueeze(0).to(DEVICE)
    )  # Keep the image in batch-channel-dimensions format
    label = (
        batch[tio.LABEL][tio.DATA][0].detach().cpu().numpy().squeeze()
    )  # Extract the label straight to a numpy array
    train_prediction = model(img).detach().cpu().numpy().squeeze().squeeze()

    # Convert the image to a 3D numpy array
    img = img.squeeze().detach().cpu().numpy().squeeze().squeeze()
    fig = plot_slices(img, train_prediction)
    fig.suptitle("Train Prediction")
    fig.tight_layout()
    fig.savefig("tmp_train_pred.png")
    plt.close(fig)

    fig = plot_slices(img, label)
    fig.suptitle("Train Truth")
    fig.tight_layout()
    fig.savefig("tmp_train_truth.png")
    plt.close(fig)

    # Plot the Dice accuracy against threshold for the validation data
    val_prediction = (
        model(
            torch.tensor(val_img, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    threshhold = plot_dice(val_prediction, val_mask)
    print(f"Best threshold from validation image: {threshhold}")

    # Get a new image and mask to evaluate on
    # This will have a transformation applied
    test_n, test_centre = 491, (1767, 277, 261)
    test_img = read_image(test_n, test_centre)
    test_mask = read_mask(test_n, test_centre)

    plot_slices(test_img, test_mask).savefig(f"{img_dir}/test_{test_n}.png")

    fig = plot_slices(test_img, None)
    fig.savefig("transformed_img.png")

    prediction = (
        model(
            torch.tensor(test_img, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    plot_roc(prediction, test_mask)

    # Threshold, using the threshhold derived from validation data
    prediction = prediction > threshhold

    # Show prediction on a plot
    fig = plot_slices(test_img, prediction)
    fig.suptitle("Prediction")
    fig.tight_layout()
    fig.savefig("tmp_pred.png")
    plt.close(fig)

    fig = plot_slices(test_img, test_mask)
    fig.suptitle("Truth")
    fig.tight_layout()
    fig.savefig("tmp_truth.png")
    plt.close(fig)

    # Plot the filters
    if not os.path.exists("filters"):
        os.mkdir("filters")
    for i, j, fig in plot_filters(model):
        path = f"filters/tmp_filter_channel_{i}_filter_{j}.png"
        print(f"Saving filter to {path}")
        plt.savefig(path)

    # Save as a tiff
    tifffile.imsave("tmp_pred.tif", prediction.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
