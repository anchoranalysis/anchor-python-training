"""Plots reconstruction of images against their original forms."""
import torch
import torch.nn as nn
from typing import Iterable
import numpy as np


from .comparison import plot_images_two_rows


def plot_reconstruction_on_first_batch(
    loader: torch.utils.data.DataLoader, model_reconstructing: nn.Module
) -> None:
    """Plots the first batch of reconstructed images against the original images.

    :param loader: a loader from which the first batch of images is loaded.
    :param model_reconstructing: a model which reconstructs an image (produces an image of identical structure to
                                 the input image, when passed an input-image).
    """

    first_batch = _retrieve_first_input_batch_from_loader(loader)

    with torch.no_grad():
        reconstruction = model_reconstructing(first_batch)

    plot_images_two_rows(
        _images_from_tensor(first_batch),
        _images_from_tensor(reconstruction),
        len(first_batch),
    )


def _retrieve_first_input_batch_from_loader(
    loader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    """Retrieves the input-images from the first batch in a loader."""
    for batch_features in loader:
        return batch_features[0]
    raise ValueError


def _images_from_tensor(images: torch.Tensor) -> Iterable[np.ndarray]:
    """Converts a tensor to an iterable of images, converting each image to an appropriately sized numpy array."""
    for index in range(images.size()[0]):
        # Convert from PyTorch RGB format (3, y, x) to Numpy expected RGB format (y, x, 3)
        yield images[index].permute(1, 2, 0).numpy()
