#!/usr/bin/env python3
"""Task 0: Implementation of a basic RNN cell"""

import numpy as np


class RNNCell:
    """A class that represents a Recurrent Neural Network (RNN) cell.

    Attributes:
        i (int): The dimensionality of the input data.
        h (int): The dimensionality of the hidden state.
        o (int): The dimensionality of the output.
        Wh (np.ndarray): The weight matrix for the hidden state.
        Wy (np.ndarray): The weight matrix for the output.
        bh (np.ndarray): The bias for the hidden state.
        by (np.ndarray): The bias for the output.
    """
    def __init__(self, i, h, o):
        """Initializes the RNNCell with random weights and zero biases."""
        self.i = i
        self.h = h
        self.o = o

        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs a forward pass for one time step of the RNN.

        Args:
            h_prev (np.ndarray): The previous hidden state.
            x_t (np.ndarray): The data input for the current time step.

        Returns:
            h_next (np.ndarray): The next hidden state.
            y (np.ndarray): The output.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
