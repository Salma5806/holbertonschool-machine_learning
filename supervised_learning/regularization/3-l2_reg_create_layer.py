#!/usr/bin/env python3
"""
Creates a TensorFlow layer that includes L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization.

    Args:
        prev: tensor, output of previous layer
        n: int, number of nodes
        activation: activation function
        lambtha: float, L2 regularization parameter

    Returns:
        The output tensor of the new layer.
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                               mode='fan_avg')
    kernel_regularizer = tf.keras.regularizers.L2(lambtha)

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )

    return dense_layer(prev)
