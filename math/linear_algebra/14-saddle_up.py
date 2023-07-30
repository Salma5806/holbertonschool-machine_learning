#!/usr/bin/env python3
"""function to multiply 2 matrices"""


import numpy as np


def np_matmul(mat1, mat2):
    """ multiply two matrices"""
    new_mat = np.dot(mat1, mat2)
    return new_mat
