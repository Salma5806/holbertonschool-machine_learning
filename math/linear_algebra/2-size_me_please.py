#!/usr/bin/env python3
""" calculates the shape of a matrix """


def matrix_shape(matrix):
    """ return the shape of a matrix """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
        