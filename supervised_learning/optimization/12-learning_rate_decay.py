#!/usr/bin/env python3
"""
Create a learning rate decay schedule using inverse time decay
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse time decay.

    Args:
        alpha (float): initial learning rate
        decay_rate (float): decay rate for learning rate
        decay_step (int): number of steps before applying decay

    Returns:
        learning_rate_schedule (tf.keras.optimizers.schedules.LearningRateSchedule):
            a learning rate schedule
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True  # ensures stepwise decay
    )
