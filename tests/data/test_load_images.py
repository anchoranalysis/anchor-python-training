"""Tests :mod:`data.load_images`"""
import pytest
from torch import utils
from typing import Tuple
from anchor_python_training import data


_IMAGE_SIZE: Tuple[int, int] = (16, 16)

_FILE_EXTENSION: str = "png"


def test_load_images(path_to_images: str) -> None:
    images = _loader(path_to_images).load_images()
    _assert_number_images(images, 15)


def test_load_images_split_two(path_to_images: str) -> None:
    train, validation = _loader(path_to_images).load_images_split_two()
    _assert_number_images(train, 10)
    _assert_number_images(validation, 5)


def test_load_images_split_three(path_to_images: str) -> None:
    train, validation, test = _loader(path_to_images).load_images_split_three()
    _assert_number_images(train, 7)
    _assert_number_images(validation, 5)
    _assert_number_images(test, 3)


def test_load_images_zero_validation_split(path_to_images: str) -> None:
    with pytest.raises(ValueError):
        _loader(path_to_images).load_images_split_two(ratio_validation=0.0)


def _assert_number_images(
    data_loader: utils.data.DataLoader, expected_number: int
) -> None:
    assert _number_images(data_loader) == expected_number


def _loader(path_to_images: str) -> data.ImageLoader:
    """Builds an image-loader."""
    return data.ImageLoader(
        path_to_images,
        [_IMAGE_SIZE, _IMAGE_SIZE],
        extension=_FILE_EXTENSION,
        rgb=True,
    )


def _number_images(data_loader: utils.data.DataLoader) -> int:
    """Number of images in a data-loader.

    See `PyTorch forum post <https://discuss.pytorch.org/t/how-can-i-know-the-size-of-data-loader-when-i-use-torchvision-datasets-imagefolder/7800/3>`_ # noqa
    """
    return len(data_loader.dataset)
