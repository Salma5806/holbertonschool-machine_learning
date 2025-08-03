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

    layer1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        padding='same', kernel_initializer=weights_init
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(layer1)
    act1 = K.layers.Activation('relu')(bn1)

    layer2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        padding='same', kernel_initializer=weights_init
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(layer2)
    act2 = K.layers.Activation('relu')(bn2)

    layer3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        padding='same', kernel_initializer=weights_init
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(layer3)

    added = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(added)

    return output
