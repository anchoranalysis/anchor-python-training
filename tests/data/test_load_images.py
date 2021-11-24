"""Tests :mod:`data.load_images`"""
from torch import utils
from typing import Tuple
from anchor_python_training import data
from anchor_python_utilities import file


_IMAGE_SIZE: Tuple[int, int] = (16, 16)

_FILE_EXTENSION: str = "png"


def test_load_images_split_two() -> None:
    train, validation = data.load_images_split_two(
        _path_to_images(),
        [_IMAGE_SIZE, _IMAGE_SIZE],
        extension=_FILE_EXTENSION,
        rgb=True,
    )
    _assert_number_images(train, 10)
    _assert_number_images(validation, 5)


def test_load_images_split_three() -> None:
    train, validation, test = data.load_images_split_three(
        _path_to_images(),
        [_IMAGE_SIZE, _IMAGE_SIZE],
        extension=_FILE_EXTENSION,
        rgb=True,
    )
    _assert_number_images(train, 7)
    _assert_number_images(validation, 5)
    _assert_number_images(test, 3)


def _assert_number_images(
    data_loader: utils.data.DataLoader, expected_number: int
) -> None:
    assert _number_images(data_loader) == expected_number


def _path_to_images() -> str:
    """The path to the images directory."""
    return file.path_same_directory(__file__, "../resources/images")


def _number_images(data_loader: utils.data.DataLoader) -> int:
    """Number of images in a data-loader.

    See `PyTorch forum post <https://discuss.pytorch.org/t/how-can-i-know-the-size-of-data-loader-when-i-use-torchvision-datasets-imagefolder/7800/3>`_ # noqa
    """
    return len(data_loader.dataset)
