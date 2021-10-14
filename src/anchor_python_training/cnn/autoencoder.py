"""An autoencoder cnn."""

import torch
import torch.nn as nn
import math
from typing import Iterable


_NUMBER_CHANNELS = 3
_NUMBER_FILTERS = 50
_KERNEL_SIZE = 3


class AutoEncoder(nn.Module):

    def __init__(self, number_channels: int, input_size: int):
        super().__init__()

        # The size we reduce all images to before flattening it into a code
        base_size = 16

        code_size = 300

        rounds_downsampling = int(math.log(input_size / base_size, 2))

        penultimate_size = _NUMBER_FILTERS*base_size*base_size

        self.encoder = nn.Sequential(
            *list(_convolution_downsampling_layers(rounds_downsampling)),
            nn.Flatten(),
            nn.Linear(penultimate_size, code_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, penultimate_size),
            nn.ReLU(True),
            nn.Unflatten(1, (_NUMBER_FILTERS,base_size,base_size)),
            *list(_convolution_upsampling_layers(rounds_downsampling)),
        )

    def forward(self, activation):
        activation = self.encoder(activation)
        activation = self.decoder(activation)
        return activation


def _convolution_downsampling_layers(number_rounds: int) -> Iterable[nn.Module]:
    """Successive convolutional layers for downsampling, each downsampled by two.
     
    The number of output channels is always :code:`_NUMBER_FILTERS`.
    
    The number of input channels is :code:`_NUMBER_FILTERS`, apart from the first layer which is :code:`_NUMBER_CHANNELS`.

    :param number_rounds: how many rounds of downsampling convolutional layers to yield.
    """
    for i in range(number_rounds):
        number_filters_in = _NUMBER_CHANNELS if i==0 else _NUMBER_FILTERS
        yield nn.Conv2d(number_filters_in, _NUMBER_FILTERS, _KERNEL_SIZE, stride=2, padding=1)
        yield nn.ReLU(True)


def _convolution_upsampling_layers(number_rounds: int) -> Iterable[nn.Module]:
    for i in range(number_rounds):
        number_filters_out = _NUMBER_CHANNELS if i==(number_rounds-1) else _NUMBER_FILTERS
        yield nn.ConvTranspose2d(_NUMBER_FILTERS, number_filters_out, _KERNEL_SIZE, stride=2, padding=1, output_padding=1)
        yield nn.ReLU(True)
