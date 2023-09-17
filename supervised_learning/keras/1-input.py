#!/usr/bin/env python3
""" builds a neural network with keras """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras
    without using the Sequential class."""
    X_input = K.layers.Input(shape=(nx,))

    X = X_input

    for i in range(len(layers)):
        X = K.layers.Dense(units=layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(X)

        if i < len(layers) - 1:
            X = K.layers.Dropout(1 - keep_prob)(X)
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
