#!/usr/bin/env python3
""" updates a variable in place using the Adam optimization algorithm """

import matplotlib.pyplot as plt
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Args:
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.
        var: The variable to be updated (numpy.ndarray).
        grad: The gradient of var (numpy.ndarray).
        v: The previous first moment of var (numpy.ndarray).
        s: The previous second moment of var (numpy.ndarray).
        t: The time step used for bias correction.

    Returns:
        The updated variable, the new first moment, and the new second moment, respectively (all as numpy.ndarrays).
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    
    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    
    return var, v, s
