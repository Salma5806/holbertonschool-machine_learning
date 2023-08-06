#!/usr/bin/env python3
""" function to calculate the derivative of a polynomial """


def poly_derivative(poly):
    """ polynomial """
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    n = len(poly)
    if n == 0:
        return None

    if n == 1:
        return [0]

    derivative = [coeff * (n - i - 1) for i, coeff in enumerate(poly[:-1])]

    if all(coeff == 0 for coeff in derivative):
        return [0]

    return derivative
