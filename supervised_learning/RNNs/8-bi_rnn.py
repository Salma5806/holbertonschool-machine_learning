#!/usr/bin/env python3
"""Task 8: Implementation of a
Bidirectional RNN"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell (BidirectionalCell): Instance of the class BidirectionalCell.
        X (np.ndarray): Data to be used, of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden state in the forward direction.
        h_t (np.ndarray): Initial hidden state in the backward direction.

    Returns:
        H (np.ndarray): Contains all of the concatenated hidden states.
        Y (np.ndarray): Contains all of the outputs.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    Hf = np.zeros((t+1, m, h))
    Hb = np.zeros((t+1, m, h))
    Hf[0] = h_0
    Hb[-1] = h_t

    for step in range(t):
        Hf[step+1] = bi_cell.forward(Hf[step], X[step])
        Hb[t-step-1] = bi_cell.backward(Hb[t-step], X[t-step-1])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
