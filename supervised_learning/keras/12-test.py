#!/usr/bin/env python3
""" tests a neural network """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network with input data and labels."""
    network.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'], optimizer='adam')
    evaluation = network.evaluate(data, labels, verbose=verbose)
    loss = evaluation[0]
    accuracy = evaluation[1]
    
    return loss, accuracy
