#!/usr/bin/env python3
"""function to perform element-wise addition, subtraction, multiplication,
and division using numpy"""


def np_elementwise(mat1, mat2):
    """ operate matrices """
    return (
        mat1 + mat2,
        mat1 - mat2,
        mat1 * mat2,
        mat1 / mat2)