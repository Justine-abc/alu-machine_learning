#!/usr/bin/env python3
"""Module for calculating polynomial derivatives."""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly: List of coefficients representing a polynomial

    Returns:
        New list of coefficients representing the derivative,
        or None if poly is invalid
    """
    # Check if poly is a valid list
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if all elements are numbers (int or float)
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # If polynomial is a constant (length 1), derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative: coefficient at index i gets multiplied by i
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    # If derivative is all zeros, return [0]
    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
