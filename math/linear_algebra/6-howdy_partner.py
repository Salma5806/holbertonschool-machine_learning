#!/usr/bin/env python3
""" function to concatenate two arrays"""


def cat_arrays(arr1, arr2):
    """ return a new list with two arrays"""
    if type(arr1) != list and type(arr2) != list:
        return None
    else:
        new_arr = arr1 + arr2
        return new_arr
