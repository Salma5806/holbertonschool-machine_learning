#!/usr/bin/env python3
""" saves an entire model and loads an entire model """

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file."""
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file."""
    loaded_model = K.models.load_model(filename)
    return loaded_model
