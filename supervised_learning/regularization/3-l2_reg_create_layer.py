#!/usr/bin/env python3
"""
Defines a function that creates a TensorFlow layer
that includes L2 Regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a TensorFlow layer that includes L2 regularization"""
    iregularizer = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=regularizer)
    return tensor(prev)
