import pytest
from anchor_python_utilities import file


@pytest.fixture()
def path_to_images() -> str:
    """The path to the images directory."""
    return file.path_same_directory(__file__, "resources/images")
