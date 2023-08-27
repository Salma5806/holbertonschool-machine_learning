#!/usr/bin/env python3
"""Defines a layer for a neural network"""



import tensorflow.compat.v1 as tf

def create_layer(prev, n, activation):
    """Creates a layer for a neural network"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init, name='layer')(prev)
    return layer
