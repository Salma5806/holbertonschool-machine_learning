#!/usr/bin/env python3
""" Task 6"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a dense layer with dropout regularization"""

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)(prev)

    if training:
        layer = tf.nn.dropout(layer, rate=1 - keep_prob)

    return layer
