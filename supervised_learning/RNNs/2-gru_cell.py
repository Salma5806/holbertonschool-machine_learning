#!/usr/bin/env python3
"""Task 2: Implementation of a
Gated Recurrent Unit Cell"""

import numpy as np


class GRUCell:
    """A class that represents a Gated Recurrent Unit Cell.

    Attributes:
        i (int): The dimensionality of the input data.
        h (int): The dimensionality of the hidden state.
        o (int): The dimensionality of the output.
        Wz (np.ndarray): The weight matrix for the update gate.
        Wr (np.ndarray): The weight matrix for the reset gate.
        Wh (np.ndarray): The weight matrix for the intermediate
        hidden state.
        Wy (np.ndarray): The weight matrix for the output.
        bz (np.ndarray): The bias for the update gate.
        br (np.ndarray): The bias for the reset gate.
        bh (np.ndarray): The bias for the intermediate hidden state.
        by (np.ndarray): The bias for the output.
    """
    def __init__(self, i, h, o):
        """Initializes the RNNCell with random weights and zero biases."""
        self.i = i
        self.h = h
        self.o = o

        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid Function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step of the GRUCell.

        Args:
        h_prev (np.ndarray): The previous hidden state.
        x_t (np.ndarray): The data input for the current time step.

        Returns:
        h_next (np.ndarray): The next hidden state.
        y (np.ndarray): The output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(concat, self.Wr) + self.br)
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_intrmed = np.tanh(np.dot(concat_r, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_intrmed
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
