"""Tests the mod:`train_autoencoder` script."""
import pathlib
import os
from anchor_python_training import train_autoencoder
from anchor_python_utilities import fixture


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
    _assert_file_exists(tmp_path, "lightning_logs/version_0/config.yaml")
    _assert_file_exists(tmp_path, "lightning_logs/version_0/hparams.yaml")
    _assert_file_exists(tmp_path, "lightning_logs/version_0/encoder.onnx")


def _assert_file_exists(root_directory: pathlib.Path, relative_path: str) -> None:
    path = os.path.join(str(root_directory), relative_path)
    assert os.path.isfile(path)
