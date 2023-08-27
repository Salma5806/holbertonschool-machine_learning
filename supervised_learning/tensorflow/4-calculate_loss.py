#!/usr/bin/env python3
"""Defines a function that calculates the softmax"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(
        y,
        l=y_pred,
    )
    return loss
