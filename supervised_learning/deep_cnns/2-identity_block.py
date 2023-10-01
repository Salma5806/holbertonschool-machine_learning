#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """identy block"""
    weights_int = K.initializers.he_normal()
    copy = A_prev
    F11, F3, F12 = filters
    layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=weights_int)(A_prev)
    bn = K.layers.BatchNormalization(axis=3)(layer1)
    activation = K.layers.Activation('relu')(bn)
    layer2 = K.layers.Conv2D(filters=F3, kernel_size=(
        3, 3), padding='same',
        kernel_initializer=weights_int)(activation)
    bn2 = K.layers.BatchNormalization(axis=3)(layer2)
    act2 = K.layers.Activation('relu')(bn2)
    layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=weights_int)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(layer3)
    fin = K.layers.Add()([bn3, copy])
    output = K.layers.Activation('relu')(fin)
    return output
