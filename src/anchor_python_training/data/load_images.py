"""Routines to load images from the file-system and split into different training, validation, test etc. datasets."""
import dataclasses

import PIL.Image
import torch
import torchvision


from typing import Tuple
from ._recursive_images_dataset import RecursiveImagesDataset


class _UnifyNumberChannels(object):
    """Transform PIL images so that they have the desired number of channels."""

    def __init__(self, rgb: bool):
        """Creates to unify to either RGB (three channels) or grayscale (single channel).

        :param rgb: true, indicates images should be converted (where necessary) to 3 channels. false to 1 channel.
        """
        self._rgb = rgb

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return image.convert("RGB" if self._rgb else "L")

    def __repr__(self):
        return self.__class__.__name__ + "()"


@dataclasses.dataclass
class ImageLoader:
    """Loads images from the file-system."""

    image_directory: str
    """The image directory to load images recursively from."""

    image_size: Tuple[int, int]
    """The size (height, width) to resize all images to."""

    extension: str = "jpg"
    """The extension (without a leading period) that all image files must match."""

    rgb: bool = True
    """When true, images are always loaded as RGB. when false, they are loaded as grayscale."""

    batch_size: int = 16
    """How many images should be in a batch."""

    num_workers: int = 1
    """The number of workers for the data-loader."""

    def load_images(self) -> torch.utils.data.DataLoader:
        """Load all images recursively from a directory.

        :returns: a data-loader with all the images that match the extension recursively in a directory.
        """
        return self._loader(self._dataset())

    def load_images_split_two(
        self,
        ratio_validation: float = 0.3,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load all images recursively from a directory, and split into *training and validation batches*.

        :param ratio_validation: a number between 0 and 1 determining linearly how many elements belong in the
                                 validation set e.g. 0.4 would try and place 40% approximately of elements into the
                                 second partition.
        :returns: the loaded images, split into training and validation data respectively.
        """
        dataset = self._dataset()

        number_first, number_second = _calculate_splits(
            len(dataset), 1.0 - ratio_validation
        )

        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [number_first, number_second],
            generator=torch.Generator(),
        )

        return self._loader(train_set), self._loader(val_set)

    def load_images_split_three(
        self,
        ratio_validation: float = 0.3,
        ratio_test: float = 0.2,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Load all images recursively from a directory, and split into *training, validation and test batches*.

        :param ratio_validation: a number between 0 and 1 determining linearly how many elements belong in the
                                 validation set e.g. 0.4 would try and place 40% approximately of elements into the
                                 second batch.
        :param ratio_test: a number between 0 and 1 determining linearly how many elements belong in the validation
                                 set e.g. 0.2 would try and place 20% approximately of elements into the third batch.
        :returns: the loaded images, split into training and validation data and test data respectively.
        """
        dataset = self._dataset()

        number_images = len(dataset)
        _, number_second = _calculate_splits(number_images, 1.0 - ratio_validation)
        _, number_third = _calculate_splits(number_images, 1.0 - ratio_test)
        number_first = number_images - number_second - number_third

        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset,
            [number_first, number_second, number_third],
            generator=torch.Generator(),
        )

        return self._loader(train_set), self._loader(val_set), self._loader(test_set)

    def _loader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _dataset(self) -> torch.utils.data.Dataset:
        """Loads the images that match the extension recursively into a dataset."""
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size),
                _UnifyNumberChannels(self.rgb),
                torchvision.transforms.ToTensor(),
            ]
        )

        return RecursiveImagesDataset(
            self.image_directory, self.extension, transform=transform
        )


def _calculate_splits(number_elements: int, ratio_split: float) -> Tuple[int, int]:
    """Calculates the size of the partitions when splitting a number of elements into two.

    :param number_elements: the total number of elements to split.
    :param ratio_split: a number between 0 and 1 determining linearly how many elements belong in the first set e.g.
                        0.6 would try and place 60% approximately of elements into the first partition.
    :returns: the number of elements in the first and second splits respectively, which are guaranteed to sum to
              :code:number_elements:.
    :throws ValueError: if either split set has zero items.
    """
    number_first = round(number_elements * ratio_split)
    number_second = number_elements - number_first

    if number_first == 0 or number_second == 0:
        raise ValueError(
            f"Each dataset must have at least one item. They had respectively {number_first} and {number_second} items."
        )

    return number_first, number_second
