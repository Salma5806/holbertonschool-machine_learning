#!/usr/bin/env python3
""" tests a neural network """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network with input data and labels."""
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
