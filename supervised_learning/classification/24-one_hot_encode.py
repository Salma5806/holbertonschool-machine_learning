#!/usr/bin/env python3
"""Task 24"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    m = Y.size
    try:
        oh_mat = np.zeros((classes, m))
        oh_mat[Y, np.arange(m)] = 1
        return oh_mat
    except Exception:
        return None
