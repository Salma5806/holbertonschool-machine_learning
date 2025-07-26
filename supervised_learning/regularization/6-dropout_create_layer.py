#!/usr/bin/env python3
"""
creating a dropout layer
its widely used for regularization
"""

import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes in the new layer
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean, whether the model is in training mode

    Returns:
        Output tensor of the new layer
    """
    # Dense layer with He initialization (VarianceScaling scale=2)
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'
    )
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    # Apply dropout
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(dense, training=training)
