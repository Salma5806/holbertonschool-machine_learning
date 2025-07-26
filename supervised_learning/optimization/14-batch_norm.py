#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function for the output

    Returns:
        A tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-7
    normalized = tf.nn.batch_normalization(z, mean, variance, beta, gamma, epsilon)
    return activation(normalized)
