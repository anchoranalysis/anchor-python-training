"""A Lightning data-module for loading images recursively from a file-system directory."""
from .load_images import load_images_split_three
import pytorch_lightning as pl


class LoadImagesModule(pl.LightningDataModule):
    """A Lightning data-module for loading images recursively from a file-system directory."""

    def __init__(
        self,
        image_directory: str,
        image_size: int,
        extension: str = "jpg",
        rgb: bool = True,
        batch_size: int = 16,
        number_workers: int = 1,
        ratio_validation: float = 0.3,
        ratio_test: float = 0.2,
    ):
        """Creates the module with necessary parameterization.

        :param image_directory: the image directory to load images recursively from.
        :param image_size: the size (height and width) to resize all images to.
        :param extension: the extension (without a leading period) that all image files must match.
        :param rgb: when true, images are always loaded as RGB. when false, they are loaded as grayscale.
        :param batch_size: how many images should be in a batch.
        :param number_workers: the number of workers for the data-loader.
        :param ratio_validation: a number between 0 and 1 determining linearly how many elements belong in the
                                 validation set e.g. 0.4 would try and place 40% approximately of elements into the
                                 second batch.
        :param ratio_test: a number between 0 and 1 determining linearly how many elements belong in the validation
                           set e.g. 0.2 would try and place 20% approximately of elements into the third batch.
        """
        super().__init__()
        (
            self.train_data,
            self.validation_data,
            self.test_data,
        ) = load_images_split_three(
            image_directory,
            [image_size, image_size],
            extension,
            rgb,
            batch_size,
            number_workers,
            ratio_validation,
            ratio_test,
        )

    def prepare_data(self) -> None:
        """Overrides :class:`pl.LightningDataModule`."""
        pass

    #
    def setup(self, stage=None):
        """Overrides :class:`pl.LightningDataModule`."""
        # Load images normally occurs here, but we've already done it in the constructor.
        pass

    def train_dataloader(self):
        """Overrides :class:`pl.LightningDataModule`."""
        return self.train_data

    def val_dataloader(self):
        """Overrides :class:`pl.LightningDataModule`."""
        return self.validation_data

    def test_dataloader(self):
        """Overrides :class:`pl.LightningDataModule`."""
        return self.test_data
