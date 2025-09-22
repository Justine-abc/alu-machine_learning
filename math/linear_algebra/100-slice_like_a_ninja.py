#!/usr/bin/env python3
"""
Slice Like A Ninja
"""

import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes
    Args:
        matrix: numpy.ndarray to slice
        axes: dictionary where key is axis and value is a tuple
              representing the slice for that axis
    Returns:
        A new numpy.ndarray sliced accordingly
    """
    # Create a list of slices (one for each dimension)
    slices = [slice(None)] * matrix.ndim

    # Replace slices for the axes provided
    for axis, s in axes.items():
        slices[axis] = slice(*s)

    # Apply the slicing
    return matrix[tuple(slices)]

