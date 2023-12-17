#!/usr/bin/env python3
"""Task 1: Forward Prop for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell (RNNCell): An instance of the RNNCell to be
        used for forward propagation.
        X (np.ndarray): The input data with shape (t, m, i).
        h_0 (np.ndarray): The initial hidden state with shape (m, h).

    Returns:
        H (np.ndarray): All hidden states with shape (t, m, h).
        Y (np.ndarray): All outputs with shape (t, m, o).
    """
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
