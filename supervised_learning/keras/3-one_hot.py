#!/usr/bin/env python3
""" converts a label vector """

import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    """
    if classes is None:
        classes = np.max(labels) + 1 
    
    one_hot_matrix = np.zeros((len(labels), classes))
    
    for i in range(len(labels)):
        one_hot_matrix[i, labels[i]] = 1
    
    return one_hot_matrix
