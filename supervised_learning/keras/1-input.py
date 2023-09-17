#!/usr/bin/env python3
""" builds a neural network with keras """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras 
    without using the Sequential class."""
    
    x = K.layers.Input(shape=(nx,))
    
    intermediate_layers = []
    
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        dense_layer = K.layers.Dense(
            nodes,
            activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha),
            name=f'dense_{i}'
        )(x if i == 0 else intermediate_layers[-1])
        
        if i < len(layers) - 1:
            dropout_layer = K.layers.Dropout(rate=1 - keep_prob, name=f'dropout_{i}')(dense_layer)
            intermediate_layers.append(dropout_layer)
        else:
            intermediate_layers.append(dense_layer)
    
    model = K.models.Model(inputs=x, outputs=intermediate_layers[-1])
    return model
