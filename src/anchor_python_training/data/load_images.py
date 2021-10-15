import torch
import torchvision
import PIL
import pathlib


from typing import Tuple


class _RecursiveImagesDataset(torch.utils.data.Dataset):
    """Finds images in a particular directory recursively."""

    def __init__(self, image_directory: str, extension: str, transform=None):
        """Creates for a particular directory and file extension.

        :param image_directory: the directory to search for images recursively in.
        :param extension: the file extension (without a leading period) which all images must end with.
        :param transform: an optional transformation that is applied to each image after it is loaded.
        """
        self._paths = list(pathlib.Path(image_directory).rglob(f"*.{extension}"))
        self._transform = transform

    def __getitem__(self, index):
        path = self._paths[index]
        image = PIL.Image.open(path)
        if self._transform is not None:
            image = self._transform(image)
        # use 1 as an arbitrary label, as all images have the same label.
        return image, 1

    def __len__(self):
        return len(self._paths)


def load_images_split(
    image_directory: str,
    image_size: Tuple[int, int],
    extension: str = "jpg",
    batch_size: int = 16,
    shuffle: bool = True,
    ratio_split: float = 0.7,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load all images recursively from a directory, and split into a training and validation batch.

    :param image_directory: the image directory to load images recursively from.
    :param image_size: the size (height, width) to resize all images to.
    :param extension: the extension (without a leading period) that all image files must match.
    :param batch_size: how many images should be in a batch.
    :param shuffle: whether to shuffle the order of images.
    :param ratio_split: a number between 0 and 1 determining linearly how many elements belong in the first set e.g.
                        0.6 would try and place 60% approximately of elements into the first partition.
    :returns: the loaded images, split into training and validation data respectively.
    """
    dataset = _dataset(image_directory, image_size, extension)

    number_first, number_second = _calculate_splits(len(dataset), ratio_split)

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [number_first, number_second],
        generator=torch.Generator().manual_seed(42),
    )

    def _loader(dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.Dataset:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )

    return _loader(train_set, True), _loader(val_set, False)


def _dataset(
    image_directory: str, image_size: Tuple[int, int], extension: str
) -> torch.utils.data.Dataset:
    """Loads the images recursively into a dataset."""
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(image_size), torchvision.transforms.ToTensor()]
    )

    return _RecursiveImagesDataset(image_directory, extension, transform=transform)


def _calculate_splits(number_elements: int, ratio_split: float) -> Tuple[int, int]:
    """Calculates the size of the partitions when splitting a number of elements into two.

    :param number_elements: the total number of elements to split.
    :param ratio_split: a number between 0 and 1 determining linearly how many elements belong in the first set e.g.
                        0.6 would try and place 60% approximately of elements into the first partition.
    :returns: the number of elements in the first and second splits respectively, which are guaranteed to sum to
              :code:number_elements:.
    """
    number_first = round(number_elements * ratio_split)
    number_second = number_elements - number_first

    if number_first == 0 or number_second == 0:
        raise RuntimeError(
            f"Each dataset must have at least one item. They had respectively {number_first} and {number_second} items."
        )

    return number_first, number_second
