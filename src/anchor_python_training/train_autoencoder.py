"""Trains (or validates etc.) an AutoEncoder model against images using the PyTorch Lightning Command Line Infterface.

Please see :class:`cnn.AutoEncoder` for details of the auto-encoder architecture.

Input Arguments
===============

Please see `PyTorch Lightning CLI <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html>`_ for
details on how to use the command line interface.

As a suggestion, first create a configuration file by saving the contents to e.g. `config.yaml`::

    python -m anchor_python_training.train_autoencoder fit --print_config

and then use these configuration file to perform the training::

    python -m anchor_python_training.train_autoencoder fit --config config.yaml

The outputted model is saved incrementally via checkpoints to a directory `lightning_logs` in the working directory
(unless otherwise configured).
"""
__author__ = "Owen Feehan"
__copyright__ = "Copyright 2021, Owen Feehan"


import torch
import torchinfo

from anchor_python_training import visualize, data, cnn
from pytorch_lightning.utilities import cli


class _MyLightningCLI(cli.LightningCLI):
    """Customized CLI implementation."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.rgb", "model.number_channels", compute_fn=_number_channels
        )
        parser.link_arguments("model.input_size", "data.image_size")

    def before_fit(self):
        """Overrides :class:`cli.LightningCLI`. Print model architecture details to the console."""

        config_fit = self.config["fit"]
        input_size = config_fit["model"]["input_size"]
        batch_size = config_fit["model"]["batch_size"]
        rgb = config_fit["data"]["rgb"]
        print(
            torchinfo.summary(
                self.model, (batch_size, _number_channels(rgb), input_size, input_size)
            )
        )
        print(self.model)

    def after_fit(self):
        """Overrides :class:`cli.LightningCLI`. Plot the first batch: reconstructed against original."""

        model = self.model.to(torch.device("cpu"))
        visualize.plot_reconstruction_on_first_batch(
            self.datamodule.validation_data, model
        )


def main():
    _MyLightningCLI(cnn.AutoEncoder, data.LoadImagesModule)


def _number_channels(rgb: bool) -> int:
    """Determines the number of channels corresponding to a RGB flag."""
    if rgb:
        return 3
    else:
        return 1


if __name__ == "__main__":
    main()
