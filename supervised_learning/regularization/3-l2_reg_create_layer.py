#!/usr/bin/env python3
"""
Creates a neural network layer in TensorFlow with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense layer with L2 regularization.

    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes in the new layer
        activation: activation function for the layer
        lambtha: float, L2 regularization parameter

    Returns:
        The output tensor of the created layer.
    """
    l2_regularizer = tf.keras.regularizers.L2(lambtha)
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        kernel_regularizer=l2_regularizer
    )
    return dense_layer(prev)
