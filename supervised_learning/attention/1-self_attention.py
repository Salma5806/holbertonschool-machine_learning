#!/usr/bin/env python3
import tensorflow as tf
"""
 task self attention
"""


class SelfAttention(tf.keras.layers.Layer):
    """Class to calculate the attention for machine translation"""
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Takes in previous decoder hidden state and outputs
            the context vector for decoder and attention weights"""
        score = self.V(tf.nn.tanh(self.W(tf.expand_dims(s_prev, 1))
                                  + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
