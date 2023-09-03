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
    layer = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=kernel_initializer,
                                  activation=None)(prev)
    gamma = tf.Variable(initial_value=tf.ones((n,)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((n,)), trainable=True)
    epsilon = 1e-8
    mean, variance = tf.nn.moments(layer, axes=[0])
    normalized = tf.nn.batch_normalization(layer, mean, variance,
                                           beta, gamma, epsilon)
    return activation(normalized)
