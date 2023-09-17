#!/usr/bin/env python3

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with Keras."""
    
    model = K.Sequential()

    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        model.add(K.layers.Dense(units=nodes,
                                activation=activation,
                                kernel_regularizer=K.regularizers.l2(lambtha),
                                input_shape=(nx if i == 0 else None)))
        
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))
    
    return model
