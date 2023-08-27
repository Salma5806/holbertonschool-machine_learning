#!/usr/bin/env python3
"""Defines a function to return two placeholders for the neural network"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """Returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
    return x, y
