#!/usr/bin/env python3
"""
Defines a function that calculates the cost of a neural network
using L2 Regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization"""
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)
