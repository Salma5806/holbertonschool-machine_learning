#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception block
    inception block as described in Going
    Deeper with Convolutions"""
    F1, F3R, F3, F5R, F5, FPP = filters
    OxO_1_input = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                                  padding='same', activation='relu')(A_prev)
    OxO_conv3x3 = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=(
        3, 3), padding='same', activation='relu')(OxO_conv3x3)
    OxO_conv5x5 = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=(
        5, 5), padding='same', activation='relu')(OxO_conv5x5)
    max_pool = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(A_prev)
    OxO_pool = K.layers.Conv2D(filters=FPP, kernel_size=(
        1, 1), padding='same', activation='relu')(max_pool)
    output = K.layers.concatenate([OxO_1_input, conv3x3, conv5x5, OxO_pool])
    return output
