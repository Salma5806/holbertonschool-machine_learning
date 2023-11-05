#!/usr/bin/env python3
"""Definiteness Module"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] < 1 or not matrix.shape[0] == matrix.shape[1]:
        return None
    if not (matrix == matrix.T).all():
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if all(ev > 0 for ev in eigenvalues):
        return ("Positive definite")
    elif all(ev >= 0 for ev in eigenvalues):
        return ("Positive semi-definite")
    elif all(ev < 0 for ev in eigenvalues):
        return ("Negative definite")
    elif all(ev <= 0 for ev in eigenvalues):
        return ("Negative semi-definite")
    else:
        return ("Indefinite")
