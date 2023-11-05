#!/usr/bin/env python3
"""Adjugate Module"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    adjugate = cofactor(matrix)
    adjugate = [[adjugate[j][i] for j in range(len(adjugate))]
                for i in range(len(adjugate[0]))]

    return adjugate
