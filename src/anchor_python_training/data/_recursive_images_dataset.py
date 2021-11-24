import PIL
import pathlib
import torch


class RecursiveImagesDataset(torch.utils.data.Dataset):
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
