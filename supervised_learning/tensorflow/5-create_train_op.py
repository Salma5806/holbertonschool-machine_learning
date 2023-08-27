#!/usr/bin/env python3
"""
Defines a function that creates the training operation
for the neural network
"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """operation that trains the network using gradient descent"""
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return (gradient_descent.minimize(loss))
