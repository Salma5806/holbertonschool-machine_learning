#!/usr/bin/env python3
""" makes a prediction using a neural network """

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes predictions using a neural network with input data."""
    predictions = network.predict(data, verbose=verbose)
    return predictions
