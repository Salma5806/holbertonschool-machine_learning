#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization in TensorFlow.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost with L2 regularization for each layer.

    Args:
        cost: tensor containing the base cost (without regularization)
        model: Keras model including layers with L2 regularization

    Returns:
        Tensor containing the total L2 cost contribution for each layer.
    """
    l2_costs = []

    # Iterate through layers and sum their regularization losses
    for layer in model.layers:
        # Each layer can have its own regularization losses
        if layer.losses:  # layer.losses contains regularization terms
            layer_l2_cost = tf.add_n(layer.losses)  # sum of L2 penalties for this layer
            l2_costs.append(layer_l2_cost)

    return tf.convert_to_tensor(l2_costs, dtype=tf.float32)
