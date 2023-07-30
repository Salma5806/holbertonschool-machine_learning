#!/usr/bin/env python3
"""  function to add two arrays element-wise """

def add_arrays(arr1, arr2):
    """ adding two arrays element wise  """

    if len(arr1) != len(arr2):
        return None
    else:
        return [i + j for i, j in zip(arr1, arr2)]

