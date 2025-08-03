#!/usr/bin/env python3
""" Identity Block """
from tensorflow import keras as K

def identity_block(A_prev, filters):
    """
    Builds an identity block as described in the paper:
    Deep Residual Learning for Image Recognition (He et al.)

    Args:
        A_prev: output from the previous layer
        filters: list or tuple of three integers [F11, F3, F12]

    Returns:
        Activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal()
    X = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)
    return X
