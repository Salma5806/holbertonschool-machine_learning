#!/usr/bin/env python3
"""  normalizes an unactivated output
of a neural network using batch normalization: """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalize an unactivated output of a
      neural network using batch normalization.

    Args:
        Z: Input data of shape (m, n).
        gamma: Scaling factors of shape (1, n).
        beta: Offsets of shape (1, n).
        epsilon: A small number used to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)
    Z_batch_normalized = gamma * Z_normalized + beta

    return Z_batch_normalized
