"""Trains a neural-network model against training-data"""

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional


def train_model(
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: Optional[int] = None
) -> None:
    """Trains a model against training-data and reports incrementally on loss.

     The loss is reported for both training-data and validation-data.

     An ADAM optimizer is used for training.

     :param train_data: the data to use for training.
     :param validation_data: the validation data to also report metrics for, at each epoch.
     :param model: the model to train, whose state will be incrementally updated each epoch.
     :param loss: the loss-function that will be minimized during training.
     :param epochs: the number of epochs to train for.
     :param learning_rate: the learning-rate to use when training.
     :param seed: an optional seed for the random-number generator, which if set, tries to make training as
                  deterministic as possible, as per `PyTorch's Reproducibility`_.

     :: _PyTorch's Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
     """
    _setup_backend(seed)

    device = _select_device()

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss_for_epoch = 0
        for batch, _ in train_data:
            loss_for_epoch += _pass_for_minibatch(batch, model, loss, optimizer, device)

        # Average loss across the number of training-samples
        loss_for_epoch = loss_for_epoch / len(train_data)

        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss_for_epoch))


def _pass_for_minibatch(batch, model: torch.nn.Module, loss: torch.nn.Module, optimizer: torch.optim.Optimizer, device) -> float:
    """Performs forward and backward passes on a particlar mini-batch."""

    batch = batch.to(device)

    optimizer.zero_grad()

    outputs = model(batch)

    train_loss = loss(outputs, batch)
    train_loss.backward()

    optimizer.step()
    return train_loss.item()


def _setup_backend(seed: Optional[int]) -> None:
    """Establishes the backend for training, including seeding the random-number generator."""
    deterministic = seed is not None

    if deterministic:
        torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = not deterministic  # as image-sizes are assumed not to vary.
    torch.backends.cudnn.deterministic = deterministic


def _select_device() -> torch.device:
    """Selects the device to execute the training on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
