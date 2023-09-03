#!/usr/bin/env python3
""" creates the training operation for a neural network in tensorflow using 
the RMSProp optimization algorithm """

import numpy as np
import tensorflow.compat.v1 as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
       loss: the loss of the network
        alpha [float]: learning rate
        beta2 [float]: RMSProp weight
        epsilon [float]: small number to avoid division by zero

    Returns:
        The updated variable and the new moment, respectively.
    """
    op = tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)

    return op