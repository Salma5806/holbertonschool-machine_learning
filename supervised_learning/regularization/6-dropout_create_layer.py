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
        training: boolean, whether in training mode (dropout applied only if True)

    Returns:
        output tensor of the new layer
    """
    # Create dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)

    # Apply dropout only during training
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    output = dropout(dense, training=training)

    return output
