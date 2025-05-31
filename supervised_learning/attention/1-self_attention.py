#!/usr/bin/env python3
"""
Self-attention layer
This module implements a self-attention layer using TensorFlow.
"""

import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    """
    Self-Attention Layer that computes a context vector using attention mechanism.
    """

    def __init__(self, units):
        """
        Initializes the self-attention layer.

        Args:
            units (int): Number of hidden units in the dense layers.
        """
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        # Print types and units
        print(type(self.W), self.W.units)
        print(type(self.U), self.U.units)
        print(type(self.V), self.V.units)

    def call(self, s_prev, hidden_states):
        """
        Compute the attention scores and return the context vector.
        """
        score = self.V(tf.nn.tanh(self.W(tf.expand_dims(s_prev, 1))
                                  + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
