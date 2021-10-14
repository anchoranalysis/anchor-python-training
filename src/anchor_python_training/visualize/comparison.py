"""Plots comparisons of a single image or set of images against nother image or set."""


import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Iterable


def plot_images_two_rows(top: Iterable[np.ndarray], bottom: Iterable[np.ndarray], row_size: int = 10, figure_size=(20, 4)) -> None:
    """Plot a top-row of images, alongside a bottom-row of images.

    :param top: the images on the top-row. There must be at least :code:row_size: elements.
    :param bottom: the images on the bottom-row. There must be at least :code:row_size: elements.
    :param row_size: the maximum number of images on a row.
    :param figure_size: the size of figure in inches, as passed to :func:`plt.figure`.
    """
    with torch.no_grad():
        plt.figure(figsize=figure_size)

        iter_top = iter(top)
        iter_bottom = iter(bottom)

        try:
            for index in range(row_size):
                _plot_image(next(iter_top), index, row_size)
                _plot_image(next(iter_bottom), index + row_size, row_size)
        except StopIteration:
            pass
        plt.show()


def _plot_image(image: np.ndarray, index: int, row_size: int) -> None:
    """Displays a row of images in a subplot."""
    ax = plt.subplot(2, row_size, index + 1)
    plt.imshow(image)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
