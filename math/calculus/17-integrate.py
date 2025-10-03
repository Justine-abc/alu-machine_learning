#!/usr/bin/env python3
"""Module for calculating polynomial integrals."""


def poly_integral(poly, c=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly: List of coefficients representing a polynomial
        c: Integration constant (default 0)

    Returns:
        New list of coefficients representing the integral,
        or None if poly or c is invalid
    """
    # Validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Validate all coefficients are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Validate c
    if not isinstance(c, (int, float)):
        return None

    # Start with the integration constant
    integral = [c]

    # Calculate integral: coefficient at index i becomes coef/(i+1) at index i+1
    for i in range(len(poly)):
        new_coef = poly[i] / (i + 1)
        # Keep as integer if it's a whole number
        if new_coef == int(new_coef):
            integral.append(int(new_coef))
        else:
            integral.append(new_coef)

    # Remove trailing zeros to make the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
