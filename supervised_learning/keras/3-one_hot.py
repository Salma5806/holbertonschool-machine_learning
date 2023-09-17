#!/usr/bin/env python3
""" converts a label vector """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    """
    if classes is None:
        classes = K.int_shape(labels)[-1]

    one_hot_matrix = K.one_hot(labels, classes)
    
    return one_hot_matrix
