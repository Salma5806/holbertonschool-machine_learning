#!/usr/bin/env python3
import tensorflow as tf
"""
self attention layer
This module implements a self-attention layer using TensorFlow.
"""

class selfAttention(tf.keras.layers.Layer):
    """
    Self-Attention Layer
    """
    def __init__(self, units):
        """
        Initialize the self-attention layer.
        """
        super(selfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Compute the attention scores and return the context vector.
        """
        w_s_prev = self.W(s_prev)
        w_s_prev_expanded = tf.expand_dims(w_s_prev, 1)
        u_hidden_states = self.U(hidden_states)
        somme = w_s_prev_expanded + u_hidden_states
        tanh_somme = tf.nn.tanh(somme)
        score = self.V(tanh_somme)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
