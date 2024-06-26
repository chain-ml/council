"""Init file."""

import os


def get_data_filename(filename: str):
    return os.path.join(os.path.dirname(__file__), "data", filename)
