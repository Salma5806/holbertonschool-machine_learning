#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization in TensorFlow.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network including L2 regularization.

    Args:
        cost: tensor containing the base cost (without regularization)
        model: Keras model that includes layers with L2 regularization

    Returns:
        A tensor containing the total cost for each layer (base + L2 penalty)
    """
    l2_costs = []
    for layer in model.layers:
        if layer.losses:  # If L2 regularization applied
            # Sum all regularization terms for this layer
            reg_term = tf.add_n(layer.losses)
            # Combine base cost + layer-specific penalty
            total_cost = cost + reg_term
            l2_costs.append(total_cost)

    return tf.convert_to_tensor(l2_costs, dtype=tf.float32)
