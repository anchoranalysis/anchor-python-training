"""Trains (or validates etc.) an AutoEncoder model against images using the PyTorch Lightning Command Line Interface.

Please see :class:`cnn.AutoEncoder` for details of the auto-encoder architecture.

Input Arguments
===============

Please see `PyTorch Lightning CLI <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html>`_ for
details on how to use the command line interface.

The help page prints all options::

    python -m anchor_python_training.train_autoencoder -h

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
import os

from anchor_python_training import visualize, data, cnn
from pytorch_lightning.utilities import cli


class _MyLightningCLI(cli.LightningCLI):
    """Customized CLI implementation."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.rgb", "model.number_channels", compute_fn=_number_channels
        )
        parser.link_arguments("model.input_size", "data.image_size")
        parser.add_argument(
            "--write.onnx.encoder", default=True, required=False, type=bool
        )
        parser.add_argument(
            "--show.reconstruction", default=True, required=False, type=bool
        )

    def before_fit(self):
        """Overrides :class:`cli.LightningCLI`. Print model architecture details to the console."""

        config_fit = self.config["fit"]
        input_size = config_fit["model"]["input_size"]
        batch_size = config_fit["data"]["batch_size"]
        rgb = config_fit["data"]["rgb"]
        print(
            torchinfo.summary(
                self.model, (batch_size, _number_channels(rgb), input_size, input_size)
            )
        )
        print(self.model)

    def after_fit(self):
        """Overrides :class:`cli.LightningCLI`. Plot the first batch: reconstructed against original."""

        # Write the encoder model as ONNX.
        if self.config["fit"]["write"]["onnx"]["encoder"]:
            _write_onnx(
                self.model,
                self.trainer.log_dir,
                "encoder",
                _extract_first_batch_from_loader(self.datamodule.train_data),
            )

        # Show examples of the first batch, input versus reconstructed.
        if self.config["fit"]["show"]["reconstruction"]:
            visualize.plot_reconstruction_on_first_batch(
                self.datamodule.validation_data, self.model.forward_encode_decode
            )


def main():
    _MyLightningCLI(cnn.AutoEncoder, data.LoadImagesModule)


def _write_onnx(
    model: torch.Tensor,
    directory: str,
    filename_without_extension: str,
    example_input: torch.Tensor,
) -> None:
    """Write a model to the file-system as ONNX.

    :param model: the model to write as ONNX.
    :param directory: the directory to write the model to.
    :param filename_without_extension: the name of the file (without the .onnx extension) to write the model to.
    :param data_loader: a data-loader, the first batch of whom is used to get an example input, as needed to write ONNX.
    """
    path = os.path.join(directory, filename_without_extension + ".onnx")
    torch.onnx.export(model, example_input, path, export_params=True)


def _extract_first_batch_from_loader(
    loader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    """Extracts the first batch from a data-loader."""
    first_batch, _ = next(iter(loader))
    return first_batch


def _number_channels(rgb: bool) -> int:
    """Determines the number of channels corresponding to a RGB flag."""
    if rgb:
        return 3
    else:
        return 1


if __name__ == "__main__":
    main()
