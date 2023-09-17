#!/usr/bin/env python3
""" trains a model """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient
      descent and optionally validates it.
    """
    if validation_data:
        val_data, val_labels = validation_data
        validation_data = (val_data, val_labels)
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
    return history
