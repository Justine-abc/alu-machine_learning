#!/usr/bin/env python3
"""Module for calculating sum of squares."""


def summation_i_squared(n):
    """
    Calculate the sum of squares from i=1 to n.

    Args:
        n: The stopping condition (upper limit)

    Returns:
        Integer value of the sum, or None if n is invalid
    """
    # Check if n is a valid number
    if not isinstance(n, (int, float)):
        return None

    # Check if n is negative
    if n < 0:
        return None

    # Convert to integer
    n = int(n)

    # Use the mathematical formula: n(n+1)(2n+1)/6
    result = n * (n + 1) * (2 * n + 1) // 6

    return result
