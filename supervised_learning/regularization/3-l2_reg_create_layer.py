#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization in TensorFlow.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates total cost including L2 regularization.

    Args:
        cost: Tensor, base cost without L2
        model: Keras model with L2 regularization

    Returns:
        Tensor with L2 regularization costs for each layer and total cost
    """
    # Collect L2 penalties from the model
    l2_losses = [tf.reduce_sum(tf.square(layer.kernel)) * layer.kernel_regularizer.l2
                 for layer in model.layers if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer]

    # Convert to TensorFlow tensor
    l2_losses = [tf.convert_to_tensor(v) for v in l2_losses]

    # Total L2 regularization
    total_l2 = tf.add_n(l2_losses) if l2_losses else 0.0

    # Return combined costs: [layer1, layer2, ..., base_cost]
    return tf.convert_to_tensor(l2_losses + [cost])
