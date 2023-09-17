#!/usr/bin/env python3
""" tests a neural network """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network with input data and labels."""
    loss, accuracy = network.evaluate(x=data,
                                      y=labels,
                                      verbose=verbose)
    return loss, accuracy