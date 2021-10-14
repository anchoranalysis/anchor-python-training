"""An autoencoder cnn."""

import torch
import torch.nn as nn
import math
from typing import Iterable, Callable


# Convolutional kernel size.
_KERNEL_SIZE = 3


# The size we reduce all images to before flattening it into a code
_BASE_SIZE = 16


# Padding, both input and output for filters.
_PADDING = 1


# Stride
_STRIDE = 2


class AutoEncoder(nn.Module):

    def __init__(self, number_channels: int, input_size: int, code_size: int = 16, number_filters: int = 16):
        super().__init__()

        self._number_channels = number_channels
        self._number_filters = number_filters

        rounds_downsampling = int(math.log(input_size / _BASE_SIZE, 2))

        penultimate_size = self._number_filters * _BASE_SIZE * _BASE_SIZE

        self.encoder = nn.Sequential(
            *_repeat_with_relu(rounds_downsampling, self._downsampling_conv_filter),
            nn.Flatten(),
            nn.Linear(penultimate_size, code_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, penultimate_size),
            nn.ReLU(True),
            nn.Unflatten(1, (self._number_filters,_BASE_SIZE,_BASE_SIZE)),
            *_repeat_with_relu(rounds_downsampling, self._upsampling_conv_filter),
        )

    def forward(self, activation):
        activation = self.encoder(activation)
        activation = self.decoder(activation)
        return activation

    def _downsampling_conv_filter(self, first: bool, last: bool) -> nn.Module:
        """A single downsampling filter.

        This is an approximately opposite counterpart to :meth:`_upsampling_conv_filter`.

        The number of output channels is always :code:`self._number_filters`.

        The number of input channels is :code:`self._number_filters`, apart from the first layer which is :code:`self._number_channels`.
        """
        number_filters_in = self._number_channels if first else self._number_filters
        return nn.Conv2d(number_filters_in, self._number_filters, _KERNEL_SIZE, stride=_STRIDE, padding=_PADDING)

    def _upsampling_conv_filter(self, first: bool, last: bool) -> nn.Module:
        """A single upsampling filter.

        This is an approximately opposite counterpart to :meth:`_downsampling_conv_filter`.

        The number of input channels is always :code:`self._number_filters`.

        The number of output channels is :code:`self._number_filters`, apart from the final layer which is :code:`self._number_channels`.
        """
        number_filters_out = self._number_channels if last else self._number_filters
        return nn.ConvTranspose2d(self._number_filters, number_filters_out, _KERNEL_SIZE, stride=_STRIDE, padding=_PADDING,
                                 output_padding=_PADDING)


def _repeat_with_relu(times: int, create_filter: Callable[[int, int], nn.Module]) -> Iterable[nn.Module]:
    """Repeats the following: a filter, followed by a ReLU for a certain number of times.

    :param times: the number of times to repeat the filter and ReLU.
    :param create_filter: creates the filter to use, passing boolean flags indicating if it is the first and last
                          iteration, respectively.
    :returns: the successive filter and ReLU modules, repeated {@code times} number of times.
    """
    for i in range(times):
        yield create_filter(i == 0, i == (times - 1))
        yield nn.ReLU(True)
