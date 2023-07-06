"""Simple container of utils."""

import numpy as np


def extract_numpy_data(arr: np.ndarray):
    """Helper to extract data from numpy array."""
    if arr.dtype.kind == 'S':  # String
        return arr.tobytes()
    elif arr.data.shape[0] > 1:  # List of vals
        return arr.tolist()
    else:
        return arr[0]  # Single value
