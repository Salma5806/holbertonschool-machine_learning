#!/usr/bin/env python3
"""
Calculates cost with L2 regularization for a Keras model.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Adds L2 regularization cost from the model's layers to the given cost.

    Args:
        cost: tensor, base cost without regularization
        model: tf.keras.Model, includes layers with L2 regularization

    Returns:
        A tensor containing:
        - regularization terms per layer
        - total regularized cost
    """
    reg_terms = []
    total_reg = 0.0

    for layer in model.layers:
        if layer.losses:  # each layer with regularizer contributes here
            layer_reg = tf.add_n(layer.losses)
            reg_terms.append(layer_reg)
            total_reg += layer_reg
        else:
            reg_terms.append(tf.constant(0.0))

    total_cost = cost + total_reg
    return tf.concat([tf.stack(reg_terms), tf.reshape(total_cost, (1,))], axis=0)
