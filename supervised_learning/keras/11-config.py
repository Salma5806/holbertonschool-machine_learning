#!/usr/bin/env python3
""" Save and Load Configuration """

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format to a file."""
    json = network.to_json()
    with open(filename, 'w+') as f:
        f.write(json)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file."""
    with open(filename, 'r') as f:
        json_string = f.read()
    model = K.models.model_from_json(json_string)
    return model
