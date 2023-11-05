#!/usr/bin/env python3
"""Cofactor Module"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor of a matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            cofactor[i][j] *= (-1) ** (i + j)

    return cofactor
