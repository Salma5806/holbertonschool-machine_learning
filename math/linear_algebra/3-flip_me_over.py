#!/usr/bin/env python3
""" check a new matrix transposed or not """


def matrix_transpose(matrix):
    """ returns the transpose of a 2D matrix """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [[row[col] for row in matrix] for col in range(len(matrix[0]))]
