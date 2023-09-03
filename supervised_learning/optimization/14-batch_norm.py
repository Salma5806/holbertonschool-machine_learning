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
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        activation=None
    )(prev)
    batch_norm_layer = tf.keras.layers.BatchNormalization()(dense_layer)
    activated_output = activation(batch_norm_layer)
    return activated_output
