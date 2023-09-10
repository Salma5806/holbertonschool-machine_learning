#!/usr/bin/env python3
"""
Defines a function that creates a TensorFlow layer
that includes L2 Regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a TensorFlow layer that includes L2 regularization"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer)
    
    return layer(prev)
