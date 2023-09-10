#!/usr/bin/env python3
"""
creates a layer of a neural network using dropout:
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout:
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)

    dropout = tf.layers.Dropout(rate=1 - keep_prob)(layer)

    return dropout
