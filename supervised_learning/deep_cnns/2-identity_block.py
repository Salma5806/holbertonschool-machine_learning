#!/usr/bin/env python3
""" Identity Block """

import tensorflow as tf
from tensorflow.keras import layers, initializers

def identity_block(A_prev, filters):
    """Builds an identity block as described in ResNet paper.

    Args:
        A_prev: Output from the previous layer (tensor).
        filters: Tuple/list of 3 integers [F11, F3, F12] representing the number
                 of filters in each convolutional layer.

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters
    initializer = initializers.HeNormal(seed=0)
    X = layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                      kernel_initializer=initializer)(A_prev)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                      kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                      kernel_initializer=initializer)(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Add()([X, A_prev])
    X = layers.Activation('relu')(X)
    return X
