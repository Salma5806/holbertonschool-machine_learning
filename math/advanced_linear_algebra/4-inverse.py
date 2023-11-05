#!/usr/bin/env python3
"""Inverse Module"""
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """Calculates the inverse of a matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None
    inverse = adjugate(matrix)
    inverse = [[inverse[i][j] / det for j in range(len(inverse))]
               for i in range(len(inverse[0]))]

    return inverse
