"""Tests :mod:`cnn.autoencoder`"""
import pytest
from torch import nn

from anchor_python_training import cnn


def test_create_valid_params():
    """Create the model with valid parameters."""
    model = cnn.AutoEncoder(1, 32)
    assert _number_parameters_model(model) == 33885


def test_create_invalid_input_size():
    """Create the model with an invalid input-size."""
    with pytest.raises(ValueError):
        cnn.AutoEncoder(1, 11)


def _number_parameters_model(model: nn.Module) -> int:
    """The number of parameters in a model (trainable or not)."""
    return sum(p.numel() for p in model.parameters())
