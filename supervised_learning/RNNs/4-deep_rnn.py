#!/usr/bin/env python3
"""Task 4: Forward Prop for a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    rnn_cells (list): A list of RNNCell instances of length l
    that will be used for the forward propagation.
        - Each RNNCell instance represents a layer in the RNN
        and should have a `forward` method for forward propagation.

    X (numpy.ndarray): Data to be used, given as a numpy.ndarray
    of shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.

    h_0 (numpy.ndarray): Initial hidden state, given as a numpy.ndarray
    of shape (l, m, h).
        - l is the number of layers.
        - m is the batch size.
        - h is the dimensionality of the hidden state.

    Returns:
    H (numpy.ndarray): A numpy.ndarray containing all of the hidden states.
    The hidden states are stored for each time step and each layer.

    Y (numpy.ndarray): A numpy.ndarray containing all of the outputs.
    The outputs are computed by the `forward` method of the RNNCell instances.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))

    H[0] = h_0

    for step in range(t):
        for layer in range(l):
            h_prev = H[step, layer]
            x_t = X[step] if layer == 0 else H[step + 1, layer - 1]

            h_next, y_next = rnn_cells[layer].forward(h_prev, x_t)

            H[step + 1, layer] = h_next

            if layer == l - 1:
                Y[step] = y_next

    return H, Y
