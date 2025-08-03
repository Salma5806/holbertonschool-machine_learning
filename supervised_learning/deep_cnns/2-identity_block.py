#!/usr/bin/env python3
""" Identity Block """
from tensorflow import keras as K

def identity_block(A_prev, filters):
    """
    Builds an identity block for a ResNet.
    
    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple of (F11, F3, F12)
        F11: filters for the first 1x1 conv
        F3: filters for the 3x3 conv
        F12: filters for the second 1x1 conv

    Returns:
    - Output tensor for the block
    """
    weights_init = K.initializers.he_normal()
    F11, F3, F12 = filters
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                        kernel_initializer=weights_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=weights_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                        kernel_initializer=weights_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)
    return X
