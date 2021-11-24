"""Tests the mod:`train_autoencoder` script."""
import pathlib
import os
from anchor_python_training import train_autoencoder
from anchor_python_utilities import fixture
from typing import Union


def test_fit(path_to_images: str, tmp_path: pathlib.Path) -> None:
    fixture.call_with_arguments(
        train_autoencoder,
        [
            "fit",
            "--data.image_directory",
            path_to_images,
            "--data.extension",
            "png",
            "--trainer.max_epochs",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--show.reconstruction",
            "false",
        ],
    )
    _assert_file_exists_in_log_dir(tmp_path, "config.yaml")
    _assert_file_exists_in_log_dir(tmp_path, "hparams.yaml")
    _assert_file_exists_in_log_dir(tmp_path, "encoder.onnx")


def test_predict(path_to_images: str) -> None:
    fixture.call_with_arguments(
        train_autoencoder,
        [
            "predict",
            "--data.image_directory",
            path_to_images,
            "--data.extension",
            "png",
        ],
    )


def _assert_file_exists_in_log_dir(directory: pathlib.Path, filename: str) -> None:
    _assert_file_exists(directory, "lightning_logs/version_0/" + filename)


def _assert_file_exists(
    directory: Union[pathlib.Path, str], relative_path: str
) -> None:
    """Asserts that a particular file exists at a path relative to a directory.

    :param directory: the root directory.
    :param relative_path: the relative path to a file (relative to :code:`directory`).
    """
    path = os.path.join(str(directory), relative_path)
    assert os.path.isfile(path)
