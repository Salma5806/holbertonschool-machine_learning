#!/usr/bin/env python3
"""  creates the training operation for a neural network in 
tensorflow using the Adam optimization algorithm """

import tensorflow.compat.v1 as tf

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow using the Adam optimization algorithm.

    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.

    Returns:
        The Adam optimization operation.
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, 
                                  beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
