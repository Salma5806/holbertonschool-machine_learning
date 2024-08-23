#!/usr/bin/env python3
"""Task 25"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    _, m = one_hot.shape
    labels = np.zeros(m, dtype=int)

    for i in range(m):
        indices = np.where(one_hot[:, i] == 1)[0]
        if len(indices) != 1:
            return None
        labels[i] = indices[0]

    return labels
