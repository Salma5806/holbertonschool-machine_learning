#!/usr/bin/env python3
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-Attention Layer for Machine Translation
    """

    def __init__(self, units):
        """
        Class constructor

        Args:
            units (int): number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Computes the context vector and attention weights

        """
        s_prev_expanded = tf.expand_dims(self.W(s_prev), 1)
        u_hidden = self.U(hidden_states)
        score = tf.nn.tanh(s_prev_expanded + u_hidden)
        score = self.V(score)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
