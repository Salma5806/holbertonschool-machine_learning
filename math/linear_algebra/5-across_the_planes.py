#!/usr/bin/env python3
"""
Task 5
"""


def add_matrices2D(mat1, mat2):
    """Add two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return (None)
    SommedMat = []
    for i in range(len(mat1)):
        SommedMat.append([])
        for j in range(len(mat1[0])):
            SommedMat[i].append(mat1[i][j] + mat2[i][j])
    return SommedMat
