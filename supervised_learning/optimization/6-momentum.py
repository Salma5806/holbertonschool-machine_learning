#!/usr/bin/env python3
"""  creates the training operation for a neural network
 in tensorflow using the gradient descent with momentum
 optimization algorithm
 """

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in TensorFlow
    using the gradient descent with momentum optimization algorithm.

    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta1: The momentum weight.

    Returns:
        The momentum optimization operation.
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)

    return train_op
