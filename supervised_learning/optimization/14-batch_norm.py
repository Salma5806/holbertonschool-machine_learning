#!/usr/bin/env python3
""" creates a batch normalization layer for a neural network in tensorflow """

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: Activated output of the previous layer.
        n: Number of nodes in the layer to be created.
        activation: Activation function to be used.

    Returns:
        A tensor of the activated output for the layer.
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=kernel_initializer, use_bias=False)
    batch_norm_layer = tf.keras.layers.BatchNormalization(epsilon=1e-8)
    output = activation(batch_norm_layer(dense_layer(prev)))

    return output