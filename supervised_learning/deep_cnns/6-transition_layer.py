#!/usr/bin/env python3
""" Task 6"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer for the DenseNet architecture"""
    initializer = K.initializers.he_normal(seed=0)
    my_layer = K.layers.BatchNormalization()(X)
    my_layer = K.layers.Activation('relu')(my_layer)

    nb_filters = int(nb_filters * compression)

    my_layer = K.layers.Conv2D(filters=nb_filters,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)
    X = K.layers.AveragePooling2D(pool_size=2,
                                  padding='same')(my_layer)

    return X, nb_filters
