#!/usr/bin/env python3
"""Determinant Module"""


def determinant(matrix, det=0):
    """Calculates the determinant of a matrix"""
    if matrix == [[]]:
        return 1

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix[0] or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    for i in range(len(matrix)):
        matrix_cp = matrix.copy()
        matrix_cp = matrix_cp[1:]

        for j in range(len(matrix_cp)):
            matrix_cp[j] = matrix_cp[j][0:i] + matrix_cp[j][i+1:]
        cofactor = (-1) ** i
        sub_det = determinant(matrix_cp)
        det += cofactor * matrix[0][i] * sub_det

    return det
