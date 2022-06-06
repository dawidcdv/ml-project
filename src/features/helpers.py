import os
import sys
from pathlib import Path


def root_dir() -> str:
    return Path(__file__).parent.parent.parent


def file_dir(file : str) -> str:
    return os.path.dirname(os.path.abspath(file))


def absolute_path(*paths : list, root : dict = root_dir()) -> str:
    return os.path.join(root, *paths)
