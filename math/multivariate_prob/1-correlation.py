#!/usr/bin/env python3
"""Correlation Module"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    correlation_matrix = np.zeros_like(C, dtype=float)

    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            correlation_matrix[i, j] = C[i, j] / \
                (np.sqrt(C[i, i]) * np.sqrt(C[j, j]))

    return correlation_matrix
