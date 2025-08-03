#!/usr/bin/env python3
"""
Building the architecture
needed to match the idea of the
Inception blocks
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block.

    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple or list of 6 integers:
        F1: number of filters in 1x1 conv
        F3R: number of filters in 1x1 conv before 3x3 conv
        F3: number of filters in 3x3 conv
        F5R: number of filters in 1x1 conv before 5x5 conv
        F5: number of filters in 5x5 conv
        FPP: number of filters in 1x1 conv after max pooling
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)

    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        padding='same', activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1),
        padding='same', activation='relu')(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5),
        padding='same', activation='relu')(conv_5x5_reduce)

    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)

    conv_pool_proj = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1),
        padding='same', activation='relu')(max_pool)

    output = K.layers.Concatenate(axis=-1)(
        [conv_1x1, conv_3x3, conv_5x5, conv_pool_proj])

    return output
