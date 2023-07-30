#!/usr/bin/env python3
""" function to add two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ adding two matrices element-wise """
    if len(mat1) != len(mat2) \
        or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    else:
        return [[i + j for i, j in zip(row1, row2)] \
                for row1, row2 in zip(mat1, mat2)]
