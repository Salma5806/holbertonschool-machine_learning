#!/usr/bin/env python3
""" Save and Load Configuration """

import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format to a file."""
    config = network.to_json()
    with open(filename, 'w') as config_file:
        json.dump(config, config_file)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file."""
    with open(filename, 'r') as config_file:
        config = json.load(config_file)
    loaded_model = K.models.model_from_json(config)
    return
