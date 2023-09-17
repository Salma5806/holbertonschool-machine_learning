#!/usr/bin/env python3
""" converts a label vector """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
