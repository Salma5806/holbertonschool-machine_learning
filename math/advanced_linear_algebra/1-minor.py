#!/usr/bin/env python3
"""Minor Module"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    det = []
    for i in range(len(matrix)):
        sub_det = []
        for j in range(len(matrix[0])):
            matrix_cp = [[cell for cell in row] for row in matrix]
            del matrix_cp[i]
            for z in range(len(matrix_cp)):
                del matrix_cp[z][j]
            sub_det.append(determinant(matrix_cp))
        det.append(sub_det)

    return det
