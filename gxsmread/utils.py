"""Simple container of utils."""

import numpy as np


def extract_numpy_data(arr: np.ndarray):
    """Helper to extract data from numpy array."""
    if arr.dtype.kind == 'S':  # String
        return arr.tobytes()
    elif len(arr.data.shape) == 0:  # Single value
        return arr.item()
    elif arr.data.shape[0] > 1:  # List of vals
        return arr.tolist()
    elif arr.data.shape[0] == 1:  # Single value in brackets
        return arr[0]
    else:
        raise TypeError('Unsure how to extract from this numpy format.')
