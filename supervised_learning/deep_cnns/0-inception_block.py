#!/usr/bin/env python3
"""
building the architecture
needed to match the idea of the
inception blocks
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:

    F1 is the number of filters in the 1x1 convolution
    F3R is the number of filters in the 1x1 convolution before the 3x3 convolution
    F3 is the number of filters in the 3x3 convolution
    F5R is the number of filters in the 1x1 convolution before the 5x5 convolution
    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution after the max pooling

    """
    C1 = K.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same',activation='relu')(A_prev)
    C11 = K.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    C12 = K.layers.Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same',activation='relu')(A_prev)
    P13 = K.layers.MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='same')(A_prev)
    C2 = K.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(C11)
    C21 = K.layers.Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same',activation='relu')(C12)
    C22 = K.layers.Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same',activation='relu')(P13)
    inception = K.layers.Concatenate(axis=-1)([C1, C2, C21, C22])
    return inception