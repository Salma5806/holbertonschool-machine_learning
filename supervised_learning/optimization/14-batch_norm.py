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
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernal_initializer=weights_initializer)
    x = layer[prev]
    gamma = tf.Variable(tf.constant(
        1, shape=(1, n), trainable=True, name="gamma"))
    beta = tf.Variable(tf.constant(
        0, shape=(1, n), trainable=True, name="gamma"))
    Z = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    return Z
