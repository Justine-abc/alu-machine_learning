#!/usr/bin/env python3
"""
Slice Like A Ninja
"""


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
    # Create a list of slices (one for each dimension of the matrix)
    slices = [slice(None)] * matrix.ndim

    # Replace slices for the axes provided
    for axis, s in axes.items():
        slices[axis] = slice(*s)

    # Apply slicing
    return matrix[tuple(slices)]
