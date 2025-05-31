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
        w_s_prev = self.W(s_prev)
        w_s_prev_expanded = tf.expand_dims(w_s_prev, 1)
        u_hidden_states = self.U(hidden_states)
        score = self.V(tf.nn.tanh(w_s_prev_expanded + u_hidden_states))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        return context_vector, attention_weights
