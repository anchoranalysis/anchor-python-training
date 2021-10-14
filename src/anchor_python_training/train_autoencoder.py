# Thanks to https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26 for inspiration.


import numpy as np

import torch
import torch.nn as nn
import torchinfo

from typing import Iterable
from anchor_python_training import visualize, data, cnn, train


def main():
    input_size = 64

    train_data, validation_data = data.load_images_split(
        r"D:\Users\owen\Pictures\To Give To Naemi 2021\Ticino Weekend (March 2021)",
        [input_size, input_size],
        "jpg",
    )

    model = cnn.AutoEncoder(number_channels=3, input_size=input_size)

    print(torchinfo.summary(model, (8, 3, input_size, input_size)))
    print(model)

    train.train_model(train_data, validation_data, model, loss=nn.MSELoss(), epochs=50)

    model = model.to(torch.device("cpu"))
    _plot_reconstruction_on_samples(validation_data, model)


def _plot_reconstruction_on_samples(
    test_loader: torch.utils.data.DataLoader,
    model_reconstructing: nn.Module,
    number_images: int = 10,
) -> None:
    """Plots a random sample of reconstructed images (from the autoencoder) against the original images."""

    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            test_examples = batch_features[0]
            reconstruction = model_reconstructing(test_examples)
            break

    visualize.plot_images_two_rows(
        _images_from_tensor(test_examples),
        _images_from_tensor(reconstruction),
        number_images,
    )


def _images_from_tensor(images: torch.Tensor) -> Iterable[np.ndarray]:
    """Converts a tensor to an iterable of images, converting each image to an appropriately sized numpy array."""
    for index in range(images.size()[0]):
        # Convert from PyTorch RGB format (3, y, x) to Numpy expected RGB format (y, x, 3)
        yield images[index].permute(1, 2, 0).numpy()


if __name__ == "__main__":
    main()
