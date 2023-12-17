#!/usr/bin/env python3
"""Task 3: Implementation of a
Long Short Term Memory Unit Cell"""

import numpy as np


class LSTMCell:
    """A class that represents a LSTM Cell.

    Attributes:
        i (int): The dimensionality of the input data.
        h (int): The dimensionality of the hidden state.
        o (int): The dimensionality of the output.
        Wf (np.ndarray): The weight matrix for the forget gate.
        Wu (np.ndarray): The weight matrix for the update gate.
        Wc (np.ndarray): The weight matrix for the intermediate
        cell state.
        Wo (np.ndarray): The weight matrix for the output gate.
        wy (np.ndarray): The weight matrix for the output.
        bf (np.ndarray): The bias for the forget gate.
        bu (np.ndarray): The bias for the update gate.
        bc (np.ndarray): The bias for the intermediate cell state.
        bo (np.ndarray): The bias for the output gate.
        by (np.ndarray): The bias for the output.
    """
    def __init__(self, i, h, o):
        """Initializes the LSTMCell with random weights and zero biases."""
        self.i = i
        self.h = h
        self.o = o

        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid Function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step of the LSTMCell.

        Args:
            h_prev (np.ndarray): The previous hidden state.
            c_prev (np.ndarray): The previous cell state.
            x_t (np.ndarray): The data input for the current time step.

        Returns:
            h_next (np.ndarray): The next hidden state.
            c_next (np.ndarray): The next cell state.
            y (np.ndarray): The output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        u_t = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_t = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_t
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
