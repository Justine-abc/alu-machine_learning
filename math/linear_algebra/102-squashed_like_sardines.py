#!/usr/bin/env python3
"""
Squashed Like Sardines
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis (supports nested lists).
    Args:
        mat1: first matrix (list of ints/floats or nested lists)
        mat2: second matrix (same shape as mat1 except along axis)
        axis: axis along which to concatenate
    Returns:
        New concatenated matrix, or None if not possible
    """
    # If concatenating scalars → only works if axis == 0
    if isinstance(mat1, (int, float)) or isinstance(mat2, (int, float)):
        return None

    # If axis = 0 → stack the lists directly
    if axis == 0:
        # Check that inner shapes are the same
        if len(mat1) == 0 or len(mat2) == 0:
            return None
        if not _same_shape(mat1[0], mat2[0]):
            return None
        return mat1 + mat2

    # Otherwise → go deeper in each row
    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        merged = cat_matrices(a, b, axis - 1)
        if merged is None:
            return None
        result.append(merged)
    return result


def _same_shape(a, b):
    """
    Helper function: check if two nested lists have the same shape
    """
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_same_shape(x, y) for x, y in zip(a, b))
    return isinstance(a, (int, float)) and isinstance(b, (int, float))
