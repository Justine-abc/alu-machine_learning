#!/usr/bin/env python3
"""
The Whole Barn
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise (supports nested lists).
    Args:
        mat1: first matrix (list of ints/floats or nested lists)
        mat2: second matrix (same shape as mat1)
    Returns:
        New matrix with element-wise sum, or None if shapes differ
    """
    # If both are numbers → add directly
    if isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    # If both are lists → check length
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for a, b in zip(mat1, mat2):
            summed = add_matrices(a, b)
            if summed is None:  # mismatch inside recursion
                return None
            result.append(summed)
        return result

    # If types mismatch → cannot add
    return None
