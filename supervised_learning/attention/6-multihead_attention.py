#!/usr/bin/env python3
"""Module implementing the Multi Head Attention layer."""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(Layer):
    """
    Multi Head Attention layer class"""

    def __init__(self, dm, h):
        """
        Class constructor"""
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = Dense(dm)
        self.Wk = Dense(dm)
        self.Wv = Dense(dm)
        self.linear = Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth) and transpose"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Forward pass for the Multi Head Attention layer"""
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
