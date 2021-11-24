"""An CNN-based `autoencoder <https://en.wikipedia.org/wiki/Autoencoder>`_ model."""

import torch.nn as nn
import math
from typing import Iterable, Callable, List, Union
import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F


BASE_IMAGE_SIZE: int = 16
"""The width and neight that images are downsampled to before flattening it into a code."""


_KERNEL_SIZE: int = 3
"""The size of each convolutional step."""


_PADDING: int = 1
"""The padding, both input and output for filters."""


_STRIDE: int = 2
"""The stride in each convolutional filter."""


class AutoEncoder(pl.LightningModule):
    """An AutoEncoder model, based upon incrementally downsampling CNNs to a flat code, and then upsampling CNNs."""

    def __init__(
        self,
        number_channels: int = 3,
        input_size: int = 32,
        code_size: int = 16,
        number_filters: int = 4,
    ):
        """Creates an AutoEncoder Model, with an architecture as described in the class-level docstring.

        The architecture is summarized as follows:

        - The input is incrementally downsampled by two until it becomes :code:`[BASE_IMAGE_SIZE, BASE_IMAGE_SIZE]`.
        - It is flattened, and there's a linear layer mapping it to a coding vector of size `code_size`.
        - It is then unflattened, and there's a linear layer mapping it back to
          :code:`[BASE_IMAGE_SIZE, BASE_IMAGE_SIZE]`.
        - Is then upsampled incrementally by two until it becomes the original input-size again.

        :param number_channels: the number of channels in the input image e.g. 1 for grayscale, 3 for RGB.
        :param input_size: the width and height of images fed into the autoencoder. It should be a power of 2 and
                           :code:`>=` :const:`BASE_IMAGE_SIZE`.
        :param code_size: how many variables exist in the 'bottleneck' latent space part of the autoencoder. Each
                          image will be encoded into a vector of this size.
        :param number_filters: the number of filters to use in each convolutional operation.
        """
        super(AutoEncoder, self).__init__()

        self._number_channels = number_channels
        self._number_filters = number_filters

        rounds_downsampling = _calculate_rounds_downsampling(input_size)

        # The size before being linear mapped into the code vector.
        penultimate_size = self._number_filters * BASE_IMAGE_SIZE * BASE_IMAGE_SIZE

        self.encoder = nn.Sequential(
            *_repeat_with_relu(rounds_downsampling, self._downsampling_conv_filter),
            nn.Flatten(),
            nn.Linear(penultimate_size, code_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, penultimate_size),
            nn.ReLU(True),
            nn.Unflatten(1, (self._number_filters, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE)),
            *_repeat_with_relu(rounds_downsampling, self._upsampling_conv_filter),
        )

    def forward(
        self, activation: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Overrides :mod:`pl.LightningModule`."""

        # Perform only the encode step when predicting.

        # When predicting, activation is both the input tensor and another.
        # This ensures only the input tensor is used
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/1137
        if isinstance(activation, List):
            activation = activation[0]

        return self.encoder(activation)

    def forward_encode_decode(
        self, input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Performs both the encode and decode step on an input (batched).

        :param input: the input tensor.
        :returns: the tensor after encoding and decoding.
        """
        temp = self.forward(input)
        return self.decoder(temp)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int):
        """Overrides :class:`pl.LightningModule`."""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        """Overrides :class:`pl.LightningModule`."""
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: List[torch.Tensor], batch_idx: int):
        """Overrides :class:`pl.LightningModule`."""
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        """Overrides :mod:`pl.LightningModule`. This docstring replaces the parent docstring which is errored."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _common_step(self, batch: List[torch.Tensor], batch_idx: int, stage: str):
        x, _ = batch
        x_hat = self.encoder(x)
        x_hat = self.decoder(x_hat)
        loss = F.mse_loss(x, x_hat)
        self.log(f"loss/{stage}", loss, on_step=True)
        return loss

    def _downsampling_conv_filter(self, first: bool, last: bool) -> nn.Module:
        """A single downsampling convolutional filter.

        This is an approximately opposite counterpart to :meth:`_upsampling_conv_filter`.

        The number of output channels is always :code:`self._number_filters`.

        The number of input channels is :code:`self._number_filters`, apart from the first layer which is
        :code:`self._number_channels`.
        """
        number_filters_in = self._number_channels if first else self._number_filters
        return nn.Conv2d(
            number_filters_in,
            self._number_filters,
            _KERNEL_SIZE,
            stride=_STRIDE,
            padding=_PADDING,
        )

    def _upsampling_conv_filter(self, first: bool, last: bool) -> nn.Module:
        """A single upsampling filter.

        This is an approximately opposite counterpart to :meth:`_downsampling_conv_filter`.

        The number of input channels is always :code:`self._number_filters`.

        The number of output channels is :code:`self._number_filters`, apart from the final layer which is
        :code:`self._number_channels`.
        """
        number_filters_out = self._number_channels if last else self._number_filters
        return nn.ConvTranspose2d(
            self._number_filters,
            number_filters_out,
            _KERNEL_SIZE,
            stride=_STRIDE,
            padding=_PADDING,
            output_padding=_PADDING,
        )


def _repeat_with_relu(
    times: int, create_filter: Callable[[int, int], nn.Module]
) -> Iterable[nn.Module]:
    """Repeats the following: a filter, followed by a ReLU for a certain number of times.

    :param times: the number of times to repeat the filter and ReLU.
    :param create_filter: creates the filter to use, passing boolean flags indicating if it is the first and last
                          iteration, respectively.
    :returns: the successive filter and ReLU modules, repeated {@code times} number of times.
    """
    for i in range(times):
        yield create_filter(i == 0, i == (times - 1))
        yield nn.ReLU(True)


def _calculate_rounds_downsampling(input_size: int) -> int:
    """Calculates how many rounds downsampling is needed so that the input becomes :const:`BASE_IMAGE_SIZE`.

    A check occurs that :code:`input_size` fulfills the necessary criteria.
    """
    rounds = math.log(input_size / BASE_IMAGE_SIZE, 2)
    if rounds.is_integer():
        return int(rounds)
    else:
        raise ValueError(
            f"input_size must be a power of 2 (>= {BASE_IMAGE_SIZE}) but is {input_size}."
        )
