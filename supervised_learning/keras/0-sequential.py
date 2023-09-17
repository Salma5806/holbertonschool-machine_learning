#!/usr/bin/env python3
""" builds a neural network """

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with Keras."""
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(lambtha),
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(lambtha)))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
