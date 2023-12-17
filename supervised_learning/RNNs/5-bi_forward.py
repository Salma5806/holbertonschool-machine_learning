#!/usr/bin/env python3
"""Task 5: Implementation of a
Bidrectional RNN cell"""

import numpy as np


class BidirectionalCell:
    """A class that represents a Bidiredtional cell of an RNN.

    Attributes:
        i (int): The dimensionality of the input data.
        h (int): The dimensionality of the hidden state.
        o (int): The dimensionality of the output.
        Whf (np.ndarray): The weight matrix for the hidden state
        in the forward direction.
        Whb (np.ndarray): The weight matrix for the hidden state
        in the backwards direction.
        Wy (np.ndarray): The weight matrix for the output.
        bhf (np.ndarray): The bias for the hidden state in the
        forwards direction.
        bhb (np.ndarray): The bias for the hidden state in the
        backward direction.
        by (np.ndarray): The bias for the output.
    """
    def __init__(self, i, h, o):
        """Initializes the  with random weights and zero biases."""
        self.i = i
        self.h = h
        self.o = o

        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs a forward pass for one time step of the RNN.

        Args:
            h_prev (np.ndarray): The previous hidden state.
            x_t (np.ndarray): The data input for the current time step.

        Returns:
            h_next (np.ndarray): The next hidden state.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next
