#!/usr/bin/env python3
""" t creates a learning rate decay operation
in tensorflow using inverse time decay """

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation
 in TensorFlow using inverse time decay.

    Args:
        alpha: The original learning rate as a TensorFlow variable.
        decay_rate: The weight used to determine
 the rate at which alpha will decay.
        global_step: The TensorFlow variable
representing the number of passes of gradient descent that have elapsed.
        decay_step: The number of passes of
gradient descent that should occur before alpha is decayed further.
    Returns:
        The learning rate decay operation as a TensorFlow variable.
    """
    updated_alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return updated_alpha
